import json
import re
import urllib.error
import urllib.parse
import urllib.request
import difflib


API_BASE = "https://api.crossref.org"
USER_AGENT = "SmartAssistantReferenceChecker/1.1"

_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)
_MARKS = re.compile(r"[\u064B-\u065F\u0670\u0640]")


def http_json(url):
    """يرجع JSON، أو None إذا كان الرمز 404 (لا يوجد سجل)."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None

        raise


def normalize_text(value):
    """تطبيع للمقارنة: يدعم العربية والأرقام العربية وكل الحروف."""
    value = str(value or "").translate(_DIGITS)
    value = _MARKS.sub("", value)
    value = value.casefold()
    value = re.sub(r"[\W_]+", " ", value, flags=re.UNICODE)

    return " ".join(value.split())


def crossref_record(record):
    titles = record.get("title") or []
    authors = record.get("author") or []
    journals = record.get("container-title") or []

    published = (
        record.get("published-print")
        or record.get("published-online")
        or record.get("issued")
        or {}
    )

    date_parts = (
        published.get("date-parts", [[None]])[0]
        if isinstance(published, dict)
        else [None]
    )

    author_names = []
    families = []

    for author in authors:
        given = author.get("given", "")
        family = author.get("family", "")

        name = " ".join(part for part in [given, family] if part)

        if name:
            author_names.append(name)

        if family:
            families.append(family)

    doi = record.get("DOI", "")

    return {
        "title": titles[0] if titles else "(العنوان غير متوفر)",
        "authors": author_names,
        "families": families,
        "year": date_parts[0] if date_parts else None,
        "journal": journals[0] if journals else "(غير متوفر)",
        "doi": doi,
        "url": (
            "https://doi.org/" + doi
            if doi
            else record.get("URL", "")
        ),
        "type": record.get("type", ""),
    }


def get_doi(text):
    text = text.strip()

    match = re.search(
        r"(?:https?://(?:dx\.)?doi\.org/|doi:\s*)"
        r"(10\.\d{4,9}/[^\s\"<>]+)",
        text,
        flags=re.I,
    )

    if not match and re.fullmatch(
        r"10\.\d{4,9}/[^\s]+",
        text,
        flags=re.I,
    ):
        return text.rstrip(".,;)]}")

    if match:
        return match.group(1).rstrip(".,;)]}")

    return None


def compare_with_claim(result, claim_text):
    """
    يقارن سجل Crossref بما كتبه المستخدم (عنوان أو استشهاد كامل).
    يرجع: title_score، year_note، author_note (قد تكون None).
    """
    claim_norm = normalize_text(claim_text)
    title_norm = normalize_text(result["title"])

    if not claim_norm or not title_norm:
        return {
            "title_score": 0.0,
            "year_note": None,
            "author_note": None,
        }

    ratio = difflib.SequenceMatcher(
        None, claim_norm, title_norm
    ).ratio()

    claim_tokens = set(claim_norm.split())
    title_tokens = set(title_norm.split())

    title_in_claim = (
        len(claim_tokens & title_tokens) / len(title_tokens)
        if title_tokens
        else 0.0
    )

    score = max(ratio, title_in_claim)

    year_note = None
    years = set(
        re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", claim_norm)
    )

    if years and result.get("year"):
        if str(result["year"]) in years:
            year_note = "ok"
        else:
            year_note = (
                "السنة المذكورة عندك ("
                + "، ".join(sorted(years))
                + ") لا تطابق سنة السجل ("
                + str(result["year"])
                + ")."
            )

    author_note = None
    families = [
        normalize_text(family)
        for family in result.get("families", [])
        if family
    ]

    looks_like_citation = len(claim_tokens) > len(title_tokens) + 2

    if families and looks_like_citation:
        matched = [
            family
            for family in families
            if family and family in claim_norm
        ]

        if matched:
            author_note = "ok"
        else:
            author_note = (
                "لم يرد في نصك اسم عائلة أي مؤلف مسجل في هذا السجل."
            )

    return {
        "title_score": score,
        "year_note": year_note,
        "author_note": author_note,
    }


def lookup_agency(doi):
    """جهة تسجيل الـ DOI بحسب Crossref (قد ترجع None)."""
    try:
        data = http_json(
            API_BASE + "/works/"
            + urllib.parse.quote(doi, safe="/")
            + "/agency"
        )
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    agency = (data.get("message") or {}).get("agency") or {}

    if isinstance(agency, dict):
        return agency.get("label") or agency.get("id")

    return None


def verify_reference(query, claimed_title=""):
    """
    يرجع dict:
      results: قائمة سجلات
      message: رسالة للمستخدم عند غياب النتائج
    """
    query = query.strip()
    claimed_title = (claimed_title or "").strip()

    if not query:
        raise ValueError("أدخل عنوان الدراسة أو DOI.")

    doi = get_doi(query)

    if doi:
        data = http_json(
            API_BASE + "/works/" + urllib.parse.quote(doi, safe="/")
        )

        if data is None:
            agency = lookup_agency(doi)
            note = (
                "لم يجد Crossref سجلًا لهذا DOI. هذا لا يثبت أنه مختلق: "
                "قد يكون مسجلًا لدى جهة أخرى (مثل DataCite)، "
                "أو لم يُفهرس بعد، أو كُتب خطأ."
            )

            if agency:
                note += " جهة التسجيل التي أبلغ بها Crossref: " + str(agency) + "."

            note += " جرّب فتح https://doi.org/" + doi + " للتأكد."

            return {"results": [], "message": note}

        record = data.get("message")

        if not isinstance(record, dict):
            raise ValueError("لم يرجع Crossref سجلًا صالحًا.")

        result = crossref_record(record)
        result["verification_type"] = "DOI"
        result["claimed"] = claimed_title

        if claimed_title:
            result["comparison"] = compare_with_claim(
                result, claimed_title
            )
            result["match_score"] = result["comparison"]["title_score"]
        else:
            result["comparison"] = None
            result["match_score"] = None

        return {"results": [result], "message": ""}

    params = urllib.parse.urlencode({
        "query.bibliographic": query,
        "rows": 5,
        "select": (
            "DOI,title,author,published-print,"
            "published-online,issued,container-title,"
            "URL,type"
        ),
    })

    data = http_json(API_BASE + "/works?" + params)

    items = (data or {}).get("message", {}).get("items", [])
    results = []

    for item in items:
        result = crossref_record(item)
        result["verification_type"] = "بحث عنوان/مرجع"
        result["claimed"] = query
        result["comparison"] = compare_with_claim(result, query)
        result["match_score"] = result["comparison"]["title_score"]
        results.append(result)

    results.sort(
        key=lambda result: result["match_score"],
        reverse=True,
    )

    return {"results": results, "message": ""}


def display_reference_result(result, st):
    score = result.get("match_score")
    comparison = result.get("comparison")
    is_doi = result["verification_type"] == "DOI"

    if is_doi and comparison is None:
        st.info(
            "عُثر على سجل لهذا DOI في Crossref، لكنك لم تُدخل العنوان "
            "الذي تدّعيه، فلم تُقارن البيانات. قارن العنوان أدناه "
            "بما تدّعيه بنفسك، فقد يكون DOI صحيحًا لورقة أخرى."
        )
    elif score is not None:
        if score >= 0.85:
            st.success(f"العنوان المسجل يطابق ما كتبته: {score:.0%}")
        elif score >= 0.60:
            st.warning(f"تطابق جزئي مع ما كتبته: {score:.0%}")
        elif is_doi:
            st.error(
                f"العنوان المسجل لا يطابق ما كتبته ({score:.0%}). "
                "قد يكون هذا DOI لورقة أخرى."
            )
        else:
            st.warning(f"التطابق ضعيف: {score:.0%}")

    if comparison:
        year_note = comparison.get("year_note")
        author_note = comparison.get("author_note")

        if year_note == "ok":
            st.write("✓ السنة المذكورة عندك تطابق سنة السجل.")
        elif year_note:
            st.warning(year_note)

        if author_note == "ok":
            st.write("✓ ورد في نصك اسم عائلة أحد المؤلفين المسجلين.")
        elif author_note:
            st.warning(author_note)

    st.markdown("**العنوان المسجل:** " + result["title"])

    authors = result["authors"]

    if authors:
        shown_authors = ", ".join(authors[:8])

        if len(authors) > 8:
            shown_authors += " وآخرون"

        st.write("**المؤلفون:** " + shown_authors)
    else:
        st.write("**المؤلفون:** غير متوفرين")

    st.write("**السنة:** " + str(result["year"] or "غير متوفرة"))
    st.write("**المجلة/المنشور:** " + result["journal"])
    st.write("**نوع السجل:** " + str(result["type"] or "غير محدد"))

    if result["doi"]:
        st.write("**DOI:** " + result["doi"])

    if result["url"]:
        st.markdown(f"[فتح سجل المصدر]({result['url']})")

    if not is_doi:
        st.caption(
            "البحث النصي في Crossref يرجع دائمًا أقرب النتائج حتى لو "
            "لم يوجد المرجع، فالنتيجة الأولى ليست إثباتًا لوجوده."
        )

    st.caption(
        "هذا الفحص يتحقق من بيانات المرجع في Crossref، "
        "ولا يثبت أن الدراسة تدعم الادعاء العلمي."
    )

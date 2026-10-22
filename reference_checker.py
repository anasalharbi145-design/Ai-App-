
import json
import re
import urllib.request
import urllib.parse
import difflib


def http_json(url):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "SmartAssistantReferenceChecker/1.0",
            "Accept": "application/json",
        },
    )

    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_text(value):
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
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

    for author in authors:
        name = " ".join(
            part
            for part in [
                author.get("given", ""),
                author.get("family", ""),
            ]
            if part
        )

        if name:
            author_names.append(name)

    doi = record.get("DOI", "")

    return {
        "title": titles[0] if titles else "(العنوان غير متوفر)",
        "authors": author_names,
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


def verify_reference(query):
    query = query.strip()

    if not query:
        raise ValueError("أدخل عنوان الدراسة أو DOI.")

    doi = get_doi(query)

    if doi:
        encoded_doi = urllib.parse.quote(doi, safe="/")

        data = http_json(
            "https://api.crossref.org/works/" + encoded_doi
        )

        record = data.get("message")

        if not isinstance(record, dict):
            raise ValueError("لم يرجع Crossref سجلًا صالحًا.")

        result = crossref_record(record)
        result["match_score"] = 1.0
        result["verification_type"] = "DOI"

        return [result]

    params = urllib.parse.urlencode({
        "query.bibliographic": query,
        "rows": 5,
        "select": (
            "DOI,title,author,published-print,"
            "published-online,issued,container-title,"
            "URL,type"
        ),
    })

    data = http_json(
        "https://api.crossref.org/works?" + params
    )

    items = data.get("message", {}).get("items", [])
    query_norm = normalize_text(query)
    results = []

    for item in items:
        result = crossref_record(item)
        title_norm = normalize_text(result["title"])

        score = (
            difflib.SequenceMatcher(
                None, query_norm, title_norm
            ).ratio()
            if query_norm and title_norm
            else 0.0
        )

        result["match_score"] = score
        result["verification_type"] = "بحث عنوان/مرجع"
        results.append(result)

    return sorted(
        results,
        key=lambda result: result["match_score"],
        reverse=True,
    )


def display_reference_result(result, st):
    score = result["match_score"]

    if result["verification_type"] == "DOI":
        st.success("عُثر على سجل لهذا DOI في Crossref.")
    elif score >= 0.85:
        st.success(f"تطابق العنوان مرتفع نسبيًا: {score:.0%}")
    elif score >= 0.60:
        st.warning(f"تطابق جزئي: {score:.0%}")
    else:
        st.warning(f"التطابق ضعيف: {score:.0%}")

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

    st.caption(
        "هذا الفحص يتحقق من بيانات المرجع في Crossref، "
        "ولا يثبت أن الدراسة تدعم الادعاء العلمي."
    )

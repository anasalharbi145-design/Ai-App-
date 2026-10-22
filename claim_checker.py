"""
تدقيق ادعاءات الإجابة: استخراج الادعاءات، البحث عن مصادر، جلب الصفحات،
مقارنة المصدر بالادعاء، ثم تصنيف النتيجة.

مبادئ:
- لا يُعتمد أي رابط ما لم تُجلب صفحته فعليًا.
- لا يُصنَّف الادعاء "مدعومًا" أو "خاطئًا" إلا إذا طابق الاقتباس
  الذي قدّمه النموذج نص الصفحة المجلوبة حرفيًا (تتحقق الشيفرة لا النموذج).
- هذا فحص آلي؛ قد يسيء النموذج فهم مصدر حقيقي. لا يغني عن مراجعة المصدر.
"""

import ipaddress
import json
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import requests


# =====================================================
# الإعدادات
# =====================================================

FOOTER_MARK = "\n\n---\n"

MAX_CLAIMS = 5
MAX_CANDIDATE_URLS = 6
MAX_USABLE_SOURCES = 3
MAX_PAGE_BYTES = 1_500_000
FETCH_TIMEOUT = 12
EXCERPT_CHARS = 2500
MIN_QUOTE_CHARS = 25
ANSWER_CHARS_LIMIT = 6000

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SmartAssistantClaimChecker/1.0)",
    "Accept": "text/html,text/plain;q=0.9,*/*;q=0.5",
}

VERDICTS = {
    "supported": ("✅", "مدعوم"),
    "partial": ("🟡", "مدعوم جزئيًا"),
    "unsupported": ("⚠️", "غير مدعوم في المصادر المفحوصة"),
    "false": ("❌", "خاطئ بحسب المصادر المفحوصة"),
    "undecided": ("❔", "غير محسوم"),
}

STOP_WORDS = {
    "the", "and", "for", "that", "with", "this", "from", "are", "was",
    "were", "has", "have", "had", "not", "but", "than", "its", "into",
    "في", "من", "على", "الى", "إلى", "عن", "هذا", "هذه", "ذلك", "التي",
    "الذي", "الذين", "كان", "كانت", "ان", "أن", "إن", "او", "أو", "ثم",
}


# =====================================================
# أدوات النص
# =====================================================

_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)
_MARKS = re.compile(r"[\u064B-\u065F\u0670\u0640]")


def norm(text):
    """تطبيع للمقارنة فقط: أرقام عربية، تشكيل، همزات، علامات ترقيم."""
    text = str(text or "").translate(_DIGITS)
    text = _MARKS.sub("", text)

    for old, new in (
        ("أ", "ا"), ("إ", "ا"), ("آ", "ا"),
        ("ى", "ي"), ("ة", "ه"),
    ):
        text = text.replace(old, new)

    text = text.casefold()
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)

    return " ".join(text.split())


def claim_terms(*texts):
    words = set()

    for text in texts:
        for word in norm(text).split():
            if word in STOP_WORDS:
                continue

            if word.isdigit() or len(word) >= 3:
                words.add(word)

    return words


def parse_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()

    try:
        return json.loads(text)
    except ValueError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except ValueError:
            return None

    return None


# =====================================================
# استدعاء Groq (عبر مكتبة groq نفسها: chat.completions)
# =====================================================

def complete(
    client,
    model,
    messages,
    tools=None,
    tool_choice=None,
    max_tokens=2048,
    effort=None,
):
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    if tools:
        kwargs["tools"] = tools

    if tool_choice:
        kwargs["tool_choice"] = tool_choice

    # reasoning_effort لنماذج gpt-oss فقط، وإن لم تقبله نسخة المكتبة نكمل بدونه.
    if effort and str(model).startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = effort

    try:
        return client.chat.completions.create(**kwargs)
    except TypeError:
        kwargs.pop("reasoning_effort", None)
        return client.chat.completions.create(**kwargs)


def message_dict(response):
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return {}

    try:
        return message.model_dump()
    except Exception:
        try:
            return dict(message)
        except Exception:
            return {}


def message_text(response):
    try:
        content = response.choices[0].message.content
        return content if isinstance(content, str) else str(content or "")
    except (AttributeError, IndexError, TypeError):
        return ""


# =====================================================
# استخراج الروابط من استجابة البحث
# =====================================================

_URL_RE = re.compile(r"https?://[^\s<>\"'\)\]]+", re.I)
_URL_KEYS = ("url", "uri", "href", "link")


def _clean_url(url):
    return str(url).strip().rstrip(".,;:)]}»")


def _collect_url_fields(value, found):
    """يقرأ حقول الروابط المسماة فقط (url/uri/href/link) مع عنوانها."""
    if isinstance(value, dict):
        title = value.get("title") or value.get("name") or ""

        for key in _URL_KEYS:
            link = value.get(key)

            if isinstance(link, dict):
                link = link.get("url") or link.get("uri")

            if isinstance(link, str) and re.match(r"^https?://", link, re.I):
                found.append({
                    "title": str(title or link),
                    "url": _clean_url(link),
                })

        for child in value.values():
            _collect_url_fields(child, found)

    elif isinstance(value, list):
        for child in value:
            _collect_url_fields(child, found)


def urls_from_message(data):
    """
    يرجع قائمة {title, url, from_tool}.
    from_tool=True: الرابط ظهر في بيانات الأداة المنفَّذة (نتائج البحث).
    from_tool=False: الرابط ذكره النموذج في نصه فقط (قد يكون مختلقًا).
    """
    tool_found = []
    _collect_url_fields(data.get("executed_tools"), tool_found)

    text_found = []
    content = data.get("content")

    if isinstance(content, str):
        for link in _URL_RE.findall(content):
            text_found.append({
                "title": _clean_url(link),
                "url": _clean_url(link),
            })

    out = []
    seen = set()

    for from_tool, items in ((True, tool_found), (False, text_found)):
        for item in items:
            if item["url"] not in seen:
                seen.add(item["url"])
                out.append({
                    "title": item["title"],
                    "url": item["url"],
                    "from_tool": from_tool,
                })

    return out


def executed_tool_names(data):
    names = []

    for tool in data.get("executed_tools") or []:
        if isinstance(tool, dict):
            names.append(
                str(tool.get("name") or tool.get("type") or "tool")
            )

    return names


# =====================================================
# جلب الصفحات بأمان
# =====================================================

def _host_is_public(hostname):
    try:
        infos = socket.getaddrinfo(hostname, None)
    except OSError:
        return False

    if not infos:
        return False

    for info in infos:
        address = info[4][0].split("%")[0]

        try:
            if not ipaddress.ip_address(address).is_global:
                return False
        except ValueError:
            return False

    return True


class _TextParser(HTMLParser):
    SKIP = {
        "script", "style", "noscript", "svg",
        "template", "nav", "footer", "iframe",
    }
    BREAK = {
        "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4",
        "h5", "h6", "section", "article", "table", "blockquote",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in self.BREAK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.parts.append(data)


def html_to_text(html):
    parser = _TextParser()

    try:
        parser.feed(html)
        parser.close()
    except Exception:
        pass

    lines = []

    for line in "".join(parser.parts).split("\n"):
        line = " ".join(line.split())

        if line:
            lines.append(line)

    return " ".join(parser.title.split()), "\n".join(lines)


def _decode(raw, content_type):
    match = re.search(r"charset=([\w-]+)", content_type or "", re.I)

    if not match:
        match = re.search(
            rb"charset=[\"']?([\w-]+)",
            raw[:4000],
            re.I,
        )
        encoding = match.group(1).decode("ascii", "ignore") if match else "utf-8"
    else:
        encoding = match.group(1)

    try:
        return raw.decode(encoding, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def fetch_page(url):
    """
    يرجع dict: ok, url, title, text, note.
    لا يتبع تحويلة إلى عنوان غير عام، ولا يقرأ أكثر من MAX_PAGE_BYTES.
    """
    current = url
    response = None

    for _ in range(4):
        parsed = urlparse(current)

        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return {"ok": False, "url": url, "note": "رابط غير صالح."}

        if not _host_is_public(parsed.hostname):
            return {
                "ok": False,
                "url": url,
                "note": "رابط غير عام؛ لم يُفتح لأسباب أمنية.",
            }

        try:
            response = requests.get(
                current,
                headers=HEADERS,
                timeout=FETCH_TIMEOUT,
                stream=True,
                allow_redirects=False,
            )
        except requests.RequestException:
            return {"ok": False, "url": url, "note": "تعذر الاتصال بالصفحة."}

        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("Location")
            response.close()

            if not location:
                return {"ok": False, "url": url, "note": "تحويلة بلا وجهة."}

            current = urljoin(current, location)
            response = None
            continue

        break

    if response is None:
        return {"ok": False, "url": url, "note": "تحويلات كثيرة."}

    try:
        if response.status_code != 200:
            return {
                "ok": False,
                "url": url,
                "note": f"رمز الاستجابة {response.status_code}.",
            }

        content_type = response.headers.get("Content-Type", "").lower()

        if "pdf" in content_type or current.lower().split("?")[0].endswith(".pdf"):
            return {
                "ok": False,
                "url": url,
                "note": "ملف PDF: الفاحص لا يقرأه حاليًا.",
            }

        if not any(
            kind in content_type
            for kind in ("html", "text/plain", "xml")
        ):
            return {
                "ok": False,
                "url": url,
                "note": "نوع المحتوى غير مدعوم للفحص.",
            }

        chunks = []
        size = 0

        for chunk in response.iter_content(65536):
            chunks.append(chunk)
            size += len(chunk)

            if size > MAX_PAGE_BYTES:
                break

        raw = b"".join(chunks)
    except requests.RequestException:
        return {"ok": False, "url": url, "note": "انقطع تحميل الصفحة."}
    finally:
        response.close()

    decoded = _decode(raw, content_type)

    if "text/plain" in content_type:
        title, text = "", decoded
    else:
        title, text = html_to_text(decoded)

    if len(text) < 200:
        return {
            "ok": False,
            "url": url,
            "note": (
                "نص الصفحة قليل جدًا؛ قد تحتاج JavaScript "
                "أو تكون خلف حاجز دخول."
            ),
        }

    return {
        "ok": True,
        "url": url,
        "title": title or url,
        "text": text,
        "note": "",
    }


def source_hint(url):
    host = (urlparse(url).hostname or "").lower()

    if "wikipedia.org" in host:
        return "موسوعة مفتوحة التحرير (ليست مصدرًا أصليًا)"

    if host.endswith("arxiv.org") or "biorxiv.org" in host or "medrxiv.org" in host:
        return "ما قبل النشر (غير محكّم غالبًا)"

    official = (
        ".gov", ".edu", ".int", ".ac.uk", "nih.gov", "who.int",
        "nature.com", "science.org", "thelancet.com", "nejm.org",
        "bmj.com", "doi.org", "sciencedirect.com", "springer.com",
        "wiley.com",
    )

    if any(host.endswith(item) or item in host for item in official):
        return "جهة رسمية/أكاديمية/ناشر علمي (مؤشر تقريبي)"

    return "غير مصنّف"


# =====================================================
# اختيار المقاطع ذات الصلة
# =====================================================

def make_excerpt(text, terms, limit=EXCERPT_CHARS):
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?؟])\s+|\n+", text)
        if len(item.strip()) >= 25
    ]

    min_score = 2 if len(terms) >= 4 else 1
    scored = []

    for index, sentence in enumerate(sentences):
        score = len(set(norm(sentence).split()) & terms)

        if score >= min_score:
            scored.append((score, index))

    if not scored:
        return ""

    scored.sort(key=lambda item: (-item[0], item[1]))

    chosen = []
    total = 0

    for score, index in scored:
        piece = sentences[index][:600]

        if total + len(piece) > limit or len(chosen) >= 8:
            break

        chosen.append(index)
        total += len(piece)

    chosen.sort()

    return "\n".join(sentences[index][:600] for index in chosen)


# =====================================================
# 1) استخراج الادعاءات
# =====================================================

EXTRACT_SYSTEM = """
أنت مستخرج ادعاءات. من النص المعطى استخرج أهم الادعاءات الواقعية القابلة
للتحقق (أرقام، تواريخ، نسب، نتائج دراسات، أحداث، تعريفات علمية).
تجاهل الآراء والنصائح والأسئلة والعبارات العامة.
اكتب كل ادعاء جملة مستقلة مفهومة بلا سياق، بلغة النص الأصلي،
وأضف query_en: صياغة إنجليزية قصيرة مناسبة للبحث عن الادعاء.
أعد JSON فقط بلا أي نص آخر وبهذا الشكل:
{"claims": [{"text": "...", "query_en": "..."}]}
"""


def extract_claims(client, model, answer, max_claims=MAX_CLAIMS):
    answer = (answer or "").strip()[:ANSWER_CHARS_LIMIT]

    if not answer:
        return []

    response = complete(
        client,
        model,
        [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"الحد الأقصى للادعاءات: {max_claims}\n\n"
                    f"النص:\n{answer}"
                ),
            },
        ],
        max_tokens=1500,
        effort="low",
    )

    data = parse_json(message_text(response))
    items = data.get("claims", []) if isinstance(data, dict) else []

    claims = []

    for item in items:
        if isinstance(item, str):
            text, query_en = item, ""
        elif isinstance(item, dict):
            text = str(item.get("text") or "")
            query_en = str(item.get("query_en") or "")
        else:
            continue

        text = text.strip()

        if len(text) >= 10:
            claims.append({
                "text": text,
                "query_en": query_en.strip(),
            })

    return claims[:max_claims]


# =====================================================
# 2) البحث عن مصادر لكل ادعاء
# =====================================================

SEARCH_SYSTEM = """
أنت باحث. استخدم أداة البحث لإيجاد مصادر أصلية أو رسمية
(ورقة محكّمة، جهة رسمية، مرجع معياري) تتناول الادعاء المعطى تحديدًا.
اكتب قائمة قصيرة: عنوان الصفحة ثم رابطها كما ظهر في نتائج البحث فقط.
لا تحكم على صحة الادعاء الآن، ولا تكتب أي رابط لم يظهر في نتائج البحث.
"""


def find_sources(client, tool_model, claim):
    query = claim.get("query_en") or claim["text"]

    response = complete(
        client,
        tool_model,
        [
            {"role": "system", "content": SEARCH_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"الادعاء: {claim['text']}\n"
                    f"صيغة بحث مقترحة: {query}"
                ),
            },
        ],
        tools=[{"type": "browser_search"}],
        tool_choice="required",
        max_tokens=2500,
        effort="low",
    )

    data = message_dict(response)

    return urls_from_message(data), bool(executed_tool_names(data))


def pick_candidates(urls, limit=MAX_CANDIDATE_URLS):
    """روابط الأداة أولًا، ومن مضيفين مختلفين قدر الإمكان."""
    ordered = sorted(urls, key=lambda item: not item["from_tool"])
    first, rest, hosts = [], [], set()

    for item in ordered:
        host = urlparse(item["url"]).hostname or ""

        if host in hosts:
            rest.append(item)
        else:
            hosts.add(host)
            first.append(item)

    return (first + rest)[:limit]


# =====================================================
# 3) الحكم على الادعاء مقابل المصادر المجلوبة
# =====================================================

JUDGE_SYSTEM = """
أنت مدقّق حقائق صارم. تحكم على الادعاء اعتمادًا على المقاطع المقدَّمة فقط،
ولا تستخدم معرفتك الخارجية.
التصنيفات:
- supported: مقطع يذكر الادعاء صراحة بتفاصيله (الأرقام، السنة، النطاق).
- partial: مقطع يدعم جزءًا من الادعاء أو بنطاق/تحفظ مختلف.
- unsupported: المقاطع المقدَّمة لا تتناول الادعاء.
- false: مقطع يناقض الادعاء صراحة.
- undecided: غموض أو تعارض بين المصادر أو نقص.
في supported وpartial وfalse يجب أن تنسخ في quote جملة واحدة من المقاطع
نسخًا حرفيًا كما وردت، وأن تحدد رقم المصدر source.
لا تغيّر في الاقتباس ولا تترجمه ولا تلخّصه.
أعد JSON فقط:
{"verdict": "...", "source": 1, "quote": "...", "reason": "جملة أو جملتان بالعربية"}
"""


def judge_claim(client, model, claim, pages):
    blocks = []

    for number, page in enumerate(pages, start=1):
        blocks.append(
            f"[مصدر {number}] {page['title']}\n"
            f"{page['url']}\n"
            f"المقاطع:\n{page['excerpt']}"
        )

    response = complete(
        client,
        model,
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"الادعاء: {claim['text']}\n\n"
                    + "\n\n".join(blocks)
                ),
            },
        ],
        max_tokens=1200,
        effort="low",
    )

    data = parse_json(message_text(response))

    return data if isinstance(data, dict) else {}


def quote_in_page(quote, page_text):
    quote_norm = norm(quote)

    if len(quote_norm) < MIN_QUOTE_CHARS:
        return False

    return quote_norm in norm(page_text)


def check_claim(client, tool_model, claim):
    result = {
        "claim": claim["text"],
        "verdict": "undecided",
        "reason": "",
        "quote": "",
        "quote_verified": False,
        "source_url": "",
        "source_title": "",
        "sources": [],
        "search_executed": False,
    }

    try:
        urls, executed = find_sources(client, tool_model, claim)
    except Exception as error:
        result["reason"] = f"تعذر البحث عن مصادر: {error}"
        return result

    result["search_executed"] = executed

    if not urls:
        result["reason"] = "لم تُرجع أداة البحث أي رابط يمكن فحصه."
        return result

    terms = claim_terms(claim["text"], claim.get("query_en", ""))
    pages = []

    for candidate in pick_candidates(urls):
        if len(pages) >= MAX_USABLE_SOURCES:
            break

        info = {
            "url": candidate["url"],
            "title": candidate["title"],
            "hint": source_hint(candidate["url"]),
            "from_tool": candidate["from_tool"],
            "status": "",
        }

        fetched = fetch_page(candidate["url"])

        if not fetched["ok"]:
            info["status"] = "لم تُفحص: " + fetched["note"]
            result["sources"].append(info)
            continue

        excerpt = make_excerpt(fetched["text"], terms)

        if not excerpt:
            info["status"] = "جُلبت الصفحة لكن لا مقطع ذا صلة بالادعاء."
            result["sources"].append(info)
            continue

        info["title"] = fetched["title"]
        info["status"] = "جُلبت وفُحصت مقاطعها ذات الصلة."
        result["sources"].append(info)

        pages.append({
            "url": fetched["url"],
            "title": fetched["title"],
            "text": fetched["text"],
            "excerpt": excerpt,
        })

    if not pages:
        result["reason"] = (
            "لم أستطع جلب أي صفحة ذات مقاطع مرتبطة بالادعاء "
            "(حاجز دفع، PDF، صفحة تعتمد JavaScript، أو لا صلة). "
            "لا أستطيع الحكم."
        )
        return result

    try:
        judged = judge_claim(client, tool_model, claim, pages)
    except Exception as error:
        result["reason"] = f"تعذر الحكم على الادعاء: {error}"
        return result

    verdict = str(judged.get("verdict", "")).strip().lower()

    if verdict not in VERDICTS:
        result["reason"] = "لم يرجع النموذج حكمًا بصيغة صالحة."
        return result

    reason = str(judged.get("reason") or "").strip()
    quote = str(judged.get("quote") or "").strip()

    try:
        source_number = int(judged.get("source", 1))
    except (TypeError, ValueError):
        source_number = 1

    if not 1 <= source_number <= len(pages):
        source_number = 1

    page = pages[source_number - 1]

    if verdict in ("supported", "partial", "false"):
        verified = quote_in_page(quote, page["text"])

        if not verified:
            # الاقتباس لم يطابق الصفحة؛ لا نقبل الحكم.
            result["reason"] = (
                "اقترح النموذج حكمًا لكن الاقتباس الذي قدّمه لم يطابق "
                "نص الصفحة المجلوبة، فلم أعتمد الحكم. "
                + (f"(ملاحظة النموذج: {reason})" if reason else "")
            )
            return result

        numbers = {
            token
            for token in norm(claim["text"]).split()
            if token.isdigit()
        }

        if (
            verdict == "supported"
            and numbers
            and not numbers & set(norm(quote).split())
        ):
            verdict = "partial"
            reason += (
                " (خُفّض الحكم إلى جزئي: الاقتباس لا يتضمن الأرقام "
                "الواردة في الادعاء.)"
            )

        result["quote"] = quote
        result["quote_verified"] = True
        result["source_url"] = page["url"]
        result["source_title"] = page["title"]

    result["verdict"] = verdict
    result["reason"] = reason

    return result


# =====================================================
# التنفيذ الكامل
# =====================================================

def audit_answer(client, model, answer, progress=None):
    """
    model: نموذج gpt-oss (يستخدم للاستخراج والبحث والحكم).
    progress: دالة اختيارية progress(i, total, text).
    """
    claims = extract_claims(client, model, answer)

    if not claims:
        return []

    results = []

    for index, claim in enumerate(claims, start=1):
        if progress:
            progress(index, len(claims), claim["text"])

        try:
            results.append(check_claim(client, model, claim))
        except Exception as error:
            results.append({
                "claim": claim["text"],
                "verdict": "undecided",
                "reason": f"خطأ غير متوقع أثناء الفحص: {error}",
                "quote": "",
                "quote_verified": False,
                "source_url": "",
                "source_title": "",
                "sources": [],
                "search_executed": False,
            })

    return results

import os
import io
import json
import base64
import re

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, UnidentifiedImageError
from groq import Groq

from reference_checker import (
    verify_reference,
    display_reference_result,
)
from claim_checker import (
    FOOTER_MARK,
    VERDICTS,
    audit_answer,
    complete as groq_complete,
    executed_tool_names,
    message_dict,
    urls_from_message,
)


# =====================================================
# 1. إعدادات التطبيق
# =====================================================

st.set_page_config(
    page_title="المساعد الذكي",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MAX_TOKENS = 4096
HISTORY_LIMIT = 16
IMAGE_MAX_SIDE = 1280
IMAGE_MAX_BYTES = 10 * 1024 * 1024

CHAT_PREFERENCES = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant",
]

VISION_PREFERENCES = [
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-27b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
]

# البحث وتشغيل الكود المدمجان مدعومان في وثائق Groq لهذين النموذجين فقط.
TOOL_MODEL_PREFERENCES = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

SYSTEM_PROMPT = """
أنت مساعد ذكاء اصطناعي عام متعدد المجالات.

قواعد الدقة:
1. لا تختلق حقائق أو أرقامًا أو دراسات أو مؤلفين أو روابط.
2. لا تدّع استخدام أداة لم تستخدمها فعلًا.
3. ميّز بين الحقيقة والاستنتاج والفرضية والرأي.
4. إذا لم تعرف، فقل ذلك بوضوح.
5. عدم العثور على دليل لا يثبت أن الادعاء خاطئ.
6. لا تخلط بين الارتباط والسببية.
7. لا تعمم نتائج دراسة واحدة دون مبرر.
8. لا توافق المستخدم لمجرد إرضائه.
9. صحح الأخطاء عند اكتشافها.
10. راجع الأرقام والوحدات والتواريخ والأسماء.

قواعد المصادر:
- وجود مرجع لا يعني أن محتواه يدعم الادعاء.
- لا تختلق عنوان دراسة أو مؤلفًا أو سنة أو رابطًا.
- فرّق بين العثور على المصدر وفحص محتواه.
- لا تقل إن مصدرًا يدعم ادعاءً إلا إذا توفر دليل على ذلك.
- وضح ما لم تتمكن من التحقق منه.
- عند تعارض الأدلة، اذكر التعارض.
- فضّل المصادر الأصلية والرسمية والدراسات المحكمة.
- لا تعرض سلسلة التفكير الداخلية؛ اعرض النتائج والأسباب المهمة.

في تدقيق الحقائق استخدم:
صحيح، صحيح جزئيًا، خاطئ، غير مدعوم، أو غير محسوم.

في البرمجة:
- لا تدّع تشغيل الكود دون تشغيله.
- حافظ على وظائف المشروع عند تعديله.
- لا تكشف مفاتيح API.
- وضّح ما لم يتم اختباره.

الأسلوب:
- أجب بلغة المستخدم.
- كن مباشرًا دون حشو.
- ابدأ بالنتيجة ثم الأدلة المهمة.
- صرّح بعدم اليقين بدل اختراع إجابة.
"""

# يُستخدم فقط عند تفعيل أداة البحث فعلًا.
RESEARCH_PROMPT = """
وضع البحث:
استخدم أداة البحث لهذا الطلب.
افحص الادعاءات الأساسية كلًا على حدة.
فضّل المصادر الأصلية والرسمية والدراسات المحكمة.
اذكر فقط الروابط التي ظهرت فعلًا في نتائج البحث.
لا تخترع روابط أو مراجع.
وجود الرابط لا يثبت أن محتواه يدعم الادعاء.
إذا لم تتمكن من فحص محتوى مصدر، فصرّح بذلك.
إذا لم تكف الأدلة، فقل إن الادعاء غير محسوم.
افصل بين صحة الفكرة وصحة تفاصيل المرجع.
"""

CODE_PROMPT = """
وضع الحساب والبرمجة:
استخدم أداة تشغيل الكود لأي حساب أو تحقق عددي بدل الحساب الذهني.
اعرض النتيجة التي أرجعتها الأداة فعلًا، ولا تدّع تشغيل كود لم يُشغَّل.
"""

MODES = {
    "متوازن": "أجب بوضوح ودقة وباختصار مناسب.",
    "تحليل عميق": """
حلل الافتراضات والبدائل والتناقضات والقيود.
اعرض النتائج والأسباب المهمة دون عرض سلسلة التفكير الداخلية.
لا تطل إذا كانت المسألة بسيطة.
""",
    "بحث وتحقق": """
افحص الادعاءات الأساسية كلًا على حدة وميّز المؤكد من غير المؤكد.
لا تدّع أنك بحثت في الإنترنت؛ البحث لا يتم إلا إذا فُعّل من الشريط الجانبي.
""",
    "معلّم خطوة بخطوة": """
اشرح الفكرة من الأساس إلى التطبيق.
استخدم خطوات واضحة وأمثلة عند الحاجة.
لا تفترض أن المستخدم يعرف المصطلحات.
""",
}


# =====================================================
# 2. تصميم الواجهة
# =====================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #101114;
        color: #f5f5f5;
    }

    [data-testid="stHeader"] {
        background: rgba(16,17,20,0.96);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 900px;
        padding: 1rem 1rem 2rem;
    }

    h1, h2, h3, p, label {
        color: #f5f5f5;
    }

    .stTextArea textarea,
    .stTextInput input {
        background: #202126 !important;
        color: #ffffff !important;
        border: 1px solid #45464d !important;
        border-radius: 12px !important;
        font-size: 16px !important;
    }

    .stButton button,
    .stFormSubmitButton button,
    .stDownloadButton button {
        border-radius: 12px;
        min-height: 44px;
        font-weight: 600;
    }

    [data-testid="stExpander"] {
        border: 1px solid #414249;
        border-radius: 12px;
    }

    a {
        overflow-wrap: anywhere;
    }

    @media (max-width: 600px) {
        [data-testid="stMainBlockContainer"] {
            padding: 0.65rem 0.7rem 1.5rem;
        }

        h1 {
            font-size: 1.6rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# 3. الاتصال بـ Groq
# =====================================================

def get_secret(name):
    value = os.getenv(name, "")

    if value:
        return value

    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


GROQ_API_KEY = get_secret("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error(
        "لم يتم العثور على GROQ_API_KEY. "
        "أضفه في Secrets بإعدادات Streamlit Cloud."
    )
    st.stop()

client = Groq(api_key=GROQ_API_KEY)


@st.cache_data(ttl=300, show_spinner=False)
def get_available_models():
    result = client.models.list()
    return sorted(model.id for model in result.data)


try:
    available_models = get_available_models()
except Exception:
    st.error(
        "تعذر جلب النماذج من Groq. "
        "تحقق من المفتاح والاتصال وحالة حسابك."
    )
    st.stop()


def filter_available(preferences):
    return [
        model
        for model in preferences
        if model in available_models
    ]


chat_options = filter_available(CHAT_PREFERENCES)
vision_options = filter_available(VISION_PREFERENCES)
tool_options = filter_available(TOOL_MODEL_PREFERENCES)

if not chat_options:
    exclusions = (
        "whisper",
        "guard",
        "speech",
        "embedding",
        "transcription",
    )

    chat_options = [
        model
        for model in available_models
        if not any(word in model.lower() for word in exclusions)
    ]

if not chat_options:
    st.error(
        "لم يظهر نموذج دردشة مناسب في حساب Groq الحالي."
    )
    st.stop()

if not vision_options:
    # لا نفترض أن نموذج النص يدعم الصور.
    vision_options = []


# =====================================================
# 4. حالة الجلسة والمحادثات
# =====================================================

DEFAULTS = {
    "messages": [],
    "last_sources": [],
    "last_model_used": "",
    "selected_chat_model": chat_options[0],
    "selected_mode": "متوازن",
    "audit_results": [],
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.session_state.selected_chat_model not in chat_options:
    st.session_state.selected_chat_model = chat_options[0]

if st.session_state.selected_mode not in MODES:
    st.session_state.selected_mode = "متوازن"


def strip_footer(text):
    """يفصل التنبيهات الملحقة بالإجابة عن نصها."""
    return text.split(FOOTER_MARK)[0]


def build_history():
    history = []

    for item in st.session_state.messages[-HISTORY_LIMIT:]:
        if (
            isinstance(item, dict)
            and item.get("role") in ("user", "assistant")
            and isinstance(item.get("content"), str)
        ):
            history.append({
                "role": item["role"],
                "content": strip_footer(item["content"]),
            })

    while history and history[0]["role"] == "assistant":
        history.pop(0)

    return history


def make_messages(
    user_text=None,
    include_user=True,
    extra_system="",
):
    mode_prompt = MODES.get(
        st.session_state.selected_mode,
        MODES["متوازن"],
    )

    system_text = SYSTEM_PROMPT + "\n\n" + mode_prompt

    if extra_system:
        system_text += "\n\n" + extra_system

    messages = [{
        "role": "system",
        "content": system_text,
    }]

    messages.extend(build_history())

    if include_user and user_text is not None:
        messages.append({
            "role": "user",
            "content": user_text,
        })

    return messages


# =====================================================
# 5. الدردشة العادية
# =====================================================

def completion_text(response):
    try:
        content = response.choices[0].message.content

        if isinstance(content, str):
            return content

        return str(content or "")
    except (AttributeError, IndexError, TypeError):
        return ""


def call_completion(model, messages):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=MAX_TOKENS,
    )


def call_chat(user_text):
    model = st.session_state.selected_chat_model

    response = call_completion(
        model,
        make_messages(user_text),
    )

    answer = completion_text(response)

    if not answer.strip():
        answer = (
            "لم يرجع النموذج إجابة نصية. "
            "حاول إعادة صياغة السؤال."
        )

    return answer, model


# =====================================================
# 6. البحث وتشغيل الكود (chat.completions عبر مكتبة groq)
# =====================================================

def research_with_web(user_text, use_search=True, use_code=False):
    """
    يستخدم أدوات Groq المدمجة (browser_search / code_interpreter)
    عبر chat.completions، وهي مدعومة في وثائق Groq لنماذج gpt-oss فقط.
    يرجع: (الإجابة، الروابط، النموذج، هل نُفِّذت أداة فعلًا)
    """
    model = st.session_state.selected_chat_model

    if model not in tool_options:
        if not tool_options:
            raise RuntimeError(
                "لا يظهر نموذج gpt-oss في حسابك، وهو النموذج المدعوم "
                "للبحث وتشغيل الكود. عطّل الأداة أو راجع حسابك في Groq."
            )

        model = tool_options[0]

    tools = []
    extra = []

    if use_search:
        tools.append({"type": "browser_search"})
        extra.append(RESEARCH_PROMPT)

    if use_code:
        tools.append({"type": "code_interpreter"})
        extra.append(CODE_PROMPT)

    messages = make_messages(
        user_text,
        extra_system="\n\n".join(extra),
    )

    response = groq_complete(
        client,
        model,
        messages,
        tools=tools,
        tool_choice="required" if use_search else "auto",
        max_tokens=MAX_TOKENS,
        effort="low" if use_search else None,
    )

    answer = completion_text(response)

    data = message_dict(response)
    tool_names = executed_tool_names(data)
    found_urls = urls_from_message(data)

    sources = [
        {"title": item["title"], "url": item["url"]}
        for item in found_urls
        if item["from_tool"]
    ][:20]

    text_only_urls = [
        item for item in found_urls if not item["from_tool"]
    ]

    if not answer.strip():
        answer = "لم يرجع النموذج إجابة نصية."

    notes = []

    if tool_names:
        notes.append(
            "✅ نُفِّذت أداة مدمجة في Groq: "
            + "، ".join(sorted(set(tool_names)))
            + f" ({len(tool_names)} استدعاء)."
        )
    else:
        notes.append(
            "⚠️ لم تسجّل الاستجابة تنفيذ أي أداة، فالإجابة من معرفة "
            "النموذج المخزنة وليست من بحث أو تشغيل فعلي."
        )

    if use_search:
        if sources:
            notes.append(
                "🔗 الروابط في القائمة أسفل المحادثة ظهرت في نتائج "
                "البحث، لكن التطبيق لم يفتح صفحاتها. استخدم «تدقيق "
                "ادعاءات آخر إجابة» في الشريط الجانبي لفحصها."
            )
        else:
            notes.append(
                "لم أستطع استخراج روابط من نتائج البحث؛ "
                "التوثيق يحتاج فحصًا يدويًا."
            )

        if text_only_urls:
            notes.append(
                "ذكر النموذج في نصه روابط لم تظهر في نتائج البحث؛ "
                "اعتبرها غير موثقة حتى تفتحها وتتأكد منها."
            )

    answer += FOOTER_MARK + "\n\n".join(notes)

    return answer, sources, model, bool(tool_names)


# =====================================================
# 7. تحليل الصور
# =====================================================

def image_to_data_url(image_file):
    image_file.seek(0)
    raw = image_file.read()

    if len(raw) > IMAGE_MAX_BYTES:
        raise ValueError("حجم الصورة يتجاوز 10 ميغابايت.")

    try:
        with Image.open(io.BytesIO(raw)) as original:
            image = original.convert("RGB")
            image.thumbnail(
                (IMAGE_MAX_SIDE, IMAGE_MAX_SIDE)
            )

            buffer = io.BytesIO()
            image.save(
                buffer,
                format="JPEG",
                quality=82,
                optimize=True,
            )

    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            "تعذر قراءة الصورة. جرّب ملف PNG أو JPG أو WEBP صالحًا."
        ) from error

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return "data:image/jpeg;base64," + encoded


def analyze_image(user_text, image_file):
    if not vision_options:
        raise RuntimeError(
            "لم يظهر نموذج صور متاح في قائمة التطبيق. "
            "راجع النماذج متعددة الوسائط المتاحة في Groq."
        )

    model = vision_options[0]
    image_url = image_to_data_url(image_file)

    prompt = user_text.strip() or (
        "حلل الصورة بدقة. افصل بين ما يظهر بوضوح "
        "وما لا تستطيع التأكد منه."
    )

    messages = make_messages(include_user=False)

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            },
        ],
    })

    response = call_completion(model, messages)
    answer = completion_text(response)

    if not answer.strip():
        answer = "لم يرجع نموذج الصور إجابة نصية."

    return answer, model


# =====================================================
# 8. مراجعة إضافية للإجابة (للإجابات بلا أدوات فقط)
# =====================================================

def review_answer(question, answer, model):
    review_prompt = """
راجع الإجابة بحثًا عن الأخطاء الواقعية والمنطقية والحسابية،
والتناقضات والمبالغة والادعاءات غير المدعومة.
صحح الأخطاء الواضحة وصرّح بما لا تستطيع حسمه.
لا تختلق مصادر أو نتائج بحث.
لا تعتبر المراجعة ضمانًا للصحة.
أعد الإجابة النهائية المصححة فقط.
"""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + "\n\n" + review_prompt,
        },
        {
            "role": "user",
            "content": (
                "السؤال:\n"
                + question
                + "\n\nالإجابة المراد مراجعتها:\n"
                + answer
            ),
        },
    ]

    response = call_completion(model, messages)
    revised = completion_text(response)

    return revised.strip() if revised.strip() else answer


# =====================================================
# 9. زر نسخ الإجابة
# =====================================================

def render_copy_button(text, button_id):
    safe_text = json.dumps(
        text,
        ensure_ascii=False,
    ).replace("</", "<\\/")

    safe_id = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        str(button_id),
    )

    components.html(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta name="viewport"
                content="width=device-width, initial-scale=1">
          <style>
            body {{
              margin: 0;
              padding: 4px 0;
              background: transparent;
              font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            }}
            button {{
              border: 1px solid #555;
              border-radius: 9px;
              background: #24262b;
              color: #fff;
              padding: 8px 14px;
              font-size: 14px;
              font-weight: 600;
            }}
            #status {{
              color: #b8c7d9;
              font-size: 12px;
              margin-right: 8px;
            }}
          </style>
        </head>
        <body>
          <button id="copy-{safe_id}" type="button">📋 نسخ الإجابة</button>
          <span id="status-{safe_id}" role="status"></span>
          <script>
            const copyText = {safe_text};
            const button = document.getElementById("copy-{safe_id}");
            const status = document.getElementById("status-{safe_id}");

            button.addEventListener("click", async () => {{
              try {{
                await navigator.clipboard.writeText(copyText);
                status.textContent = "تم النسخ ✓";
              }} catch (error) {{
                status.textContent = "تعذر النسخ التلقائي.";
              }}
            }});
          </script>
        </body>
        </html>
        """,
        height=48,
        scrolling=False,
    )


# =====================================================
# 10. عرض نتائج تدقيق الادعاءات
# =====================================================

def safe_md(text, limit=300):
    text = " ".join(str(text or "").split())
    return text.replace("[", "(").replace("]", ")")[:limit]


def render_audit(results):
    st.subheader("🔬 نتيجة تدقيق الادعاءات")

    st.warning(
        "هذا فحص آلي وليس ضمانًا. جلبت الشيفرة الصفحات فعليًا وتحققت "
        "من أن الاقتباس يطابق نص الصفحة حرفيًا، لكن فهم النموذج لمعنى "
        "المصدر قد يخطئ. افتح المصدر قبل الاعتماد عليه."
    )

    for number, item in enumerate(results, start=1):
        icon, label = VERDICTS.get(item["verdict"], VERDICTS["undecided"])
        st.markdown(f"{icon} **{number}.** {safe_md(item['claim'], 120)}")

    for number, item in enumerate(results, start=1):
        icon, label = VERDICTS.get(item["verdict"], VERDICTS["undecided"])

        with st.expander(f"{icon} الادعاء {number}: {label}"):
            st.write(item["claim"])

            if item.get("reason"):
                st.write("**السبب:** " + item["reason"])

            if item.get("quote"):
                st.markdown("**الاقتباس من المصدر:**")
                st.markdown("> " + safe_md(item["quote"], 600))

                if item.get("quote_verified"):
                    st.caption("✓ طابق الاقتباس نص الصفحة المجلوبة.")

            if item.get("source_url"):
                st.markdown(
                    f"[{safe_md(item.get('source_title') or 'المصدر')}]"
                    f"({item['source_url']})"
                )

            if item["verdict"] == "unsupported":
                st.caption(
                    "«غير مدعوم» تعني أن المقاطع المفحوصة لا تتناول "
                    "الادعاء؛ لا تعني أنه خاطئ."
                )

            if item.get("sources"):
                st.caption("الصفحات التي حاولت فحصها:")

                for source in item["sources"]:
                    origin = (
                        "من نتائج البحث"
                        if source.get("from_tool")
                        else "ذكرها النموذج في نصه"
                    )
                    st.markdown(
                        f"- {safe_md(source.get('title'), 80)} — "
                        f"{source.get('status', '')} "
                        f"({origin}؛ {source.get('hint', '')}) "
                        f"{source['url']}"
                    )


# =====================================================
# 11. الشريط الجانبي
# =====================================================

with st.sidebar:
    st.header("⚙️ إعدادات المساعد")

    st.session_state.selected_chat_model = st.selectbox(
        "نموذج الدردشة",
        options=chat_options,
        index=chat_options.index(
            st.session_state.selected_chat_model
        ),
    )

    st.session_state.selected_mode = st.selectbox(
        "طريقة الإجابة",
        options=list(MODES.keys()),
        index=list(MODES.keys()).index(
            st.session_state.selected_mode
        ),
    )

    if vision_options:
        st.caption("نموذج الصور: " + vision_options[0])
    else:
        st.warning("لم يظهر نموذج صور ضمن قائمة النماذج المحددة.")

    st.divider()
    st.subheader("🧰 الأدوات")

    if tool_options:
        st.caption(
            "البحث وتشغيل الكود يعملان على: " + "، ".join(tool_options)
        )
    else:
        st.warning(
            "لا يظهر نموذج gpt-oss في حسابك؛ "
            "البحث وتشغيل الكود غير متاحين."
        )

    enable_web_search = st.toggle(
        "🔎 البحث في الإنترنت",
        value=False,
        help="أداة browser_search المدمجة في Groq (نماذج gpt-oss).",
    )

    enable_code = st.toggle(
        "🧮 تشغيل الكود والحسابات",
        value=False,
        help="أداة code_interpreter المدمجة في Groq (نماذج gpt-oss).",
    )

    enable_review = st.toggle(
        "🛡️ مراجعة إضافية للإجابة",
        value=False,
        help=(
            "طلب إضافي يراجع فيه النموذج إجابته من معرفته، ولا يضمن "
            "خلوها من الأخطاء. لا تُطبَّق على إجابات البحث أو الكود."
        ),
    )

    st.divider()
    st.subheader("🔬 تدقيق ادعاءات آخر إجابة")

    st.caption(
        "يستخرج الادعاءات، يبحث عن مصادر، يجلب الصفحات فعليًا، ويقارن "
        "المصدر بالادعاء. يستهلك عدة طلبات (حتى 5 ادعاءات). "
        "فحص آلي وليس ضمانًا."
    )

    last_answer = ""

    for item in reversed(st.session_state.messages):
        if item["role"] == "assistant":
            last_answer = strip_footer(item["content"])
            break

    audit_clicked = st.button(
        "دقّق آخر إجابة",
        use_container_width=True,
        disabled=(not last_answer) or (not tool_options),
    )

    if audit_clicked:
        progress_slot = st.empty()

        def show_progress(index, total, text):
            progress_slot.caption(
                f"جارٍ فحص الادعاء {index} من {total}..."
            )

        try:
            with st.spinner("جارٍ تدقيق الادعاءات (قد يستغرق دقائق)..."):
                st.session_state.audit_results = audit_answer(
                    client,
                    tool_options[0],
                    last_answer,
                    progress=show_progress,
                )

            progress_slot.empty()

            if not st.session_state.audit_results:
                st.info("لم أجد في الإجابة ادعاءات واقعية قابلة للفحص.")

        except Exception as error:
            progress_slot.empty()
            st.error("تعذر تدقيق الإجابة.")
            st.caption(str(error))

    st.divider()
    st.subheader("📚 التحقق من مرجع علمي")

    st.caption(
        "يفحص بيانات المرجع في Crossref، "
        "ولا يثبت أن الدراسة تدعم الادعاء العلمي."
    )

    with st.form("reference_verification_form"):
        reference_query = st.text_input(
            "عنوان الدراسة أو DOI",
            placeholder="مثال: 10.xxxx/xxxxx",
        )

        reference_claimed = st.text_input(
            "العنوان أو الاستشهاد الذي تدّعيه (عند إدخال DOI)",
            placeholder="اكتب عنوان الدراسة كما ذُكر لك",
        )

        verify_submitted = st.form_submit_button(
            "تحقق من المرجع",
            use_container_width=True,
        )

    if verify_submitted:
        try:
            with st.spinner("جارٍ البحث في Crossref..."):
                outcome = verify_reference(
                    reference_query,
                    reference_claimed,
                )

            results = outcome["results"]

            if not results:
                st.warning(
                    outcome["message"]
                    or (
                        "لم تظهر نتائج في Crossref. "
                        "هذا لا يثبت أن المرجع مختلق."
                    )
                )
            else:
                for index, result in enumerate(results):
                    with st.expander(
                        f"النتيجة {index + 1}: {result['title']}",
                        expanded=(index == 0),
                    ):
                        display_reference_result(result, st)

        except Exception as error:
            st.error("تعذر التحقق من المرجع.")
            st.caption(str(error))

    st.divider()
    st.subheader("💾 إدارة المحادثات")

    history_json = json.dumps(
        st.session_state.messages,
        ensure_ascii=False,
        indent=2,
    )

    st.download_button(
        "حفظ المحادثة",
        data=history_json,
        file_name="chat_history.json",
        mime="application/json",
        use_container_width=True,
    )

    uploaded_history = st.file_uploader(
        "استعادة محادثة",
        type=["json"],
        key="history_import",
    )

    if st.button("استعادة الملف", use_container_width=True):
        if uploaded_history is None:
            st.warning("ارفع ملف المحادثة أولًا.")
        else:
            try:
                data = json.load(uploaded_history)

                if not isinstance(data, list):
                    raise ValueError("تنسيق الملف غير صحيح.")

                valid_messages = []

                for item in data:
                    if (
                        isinstance(item, dict)
                        and item.get("role") in ("user", "assistant")
                        and isinstance(item.get("content"), str)
                    ):
                        valid_messages.append({
                            "role": item["role"],
                            "content": item["content"],
                        })

                st.session_state.messages = valid_messages
                st.session_state.last_sources = []
                st.session_state.last_model_used = ""
                st.session_state.audit_results = []

                st.success("تمت استعادة سجل المحادثة.")
                st.rerun()

            except Exception:
                st.error(
                    "تعذرت استعادة الملف. تأكد من سلامة JSON."
                )

    if st.button("مسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.last_model_used = ""
        st.session_state.audit_results = []
        st.rerun()


# =====================================================
# 12. واجهة المساعد
# =====================================================

st.title("🤖 المساعد الذكي الشامل")

st.caption(
    "دردشة • بحث • صور • تعليم • برمجة • تدقيق مراجع"
)

if not st.session_state.messages:
    st.info(
        "اكتب سؤالك. استخدم البحث للمعلومات الحديثة، "
        "وقسم التحقق لفحص بيانات مرجع علمي مستقلًا."
    )

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_copy_button(
                strip_footer(message["content"]),
                f"history_{index}",
            )

if st.session_state.last_sources:
    with st.expander("🔗 الروابط التي ظهرت في آخر بحث"):
        st.warning(
            "وجود الرابط لا يثبت أنه يدعم الادعاء. "
            "افتح المصدر وراجع محتواه."
        )

        for source in st.session_state.last_sources:
            st.markdown(
                f"- [{safe_md(source['title'], 120)}]({source['url']})"
            )

if st.session_state.last_model_used:
    st.caption(
        "النموذج المستخدم في آخر إجابة: "
        + st.session_state.last_model_used
    )

if st.session_state.audit_results:
    render_audit(st.session_state.audit_results)


# =====================================================
# 13. إدخال السؤال والصورة
# =====================================================

with st.form("chat_form", clear_on_submit=True):
    user_text = st.text_area(
        "اكتب سؤالك",
        placeholder="اسأل عن أي موضوع...",
        height=115,
    )

    uploaded_image = st.file_uploader(
        "أرفق صورة (اختياري)",
        type=["png", "jpg", "jpeg", "webp"],
    )

    submitted = st.form_submit_button(
        "إرسال",
        type="primary",
        use_container_width=True,
    )


# =====================================================
# 14. تنفيذ الطلب
# =====================================================

if submitted:
    if not user_text.strip() and uploaded_image is None:
        st.warning("اكتب سؤالًا أو أرفق صورة أولًا.")

    else:
        question = user_text.strip()
        saved_question = question

        if uploaded_image is not None:
            saved_question = (
                (question + "\n\n") if question else ""
            ) + "[أرفق المستخدم صورة]"

        try:
            sources = []
            model_used = ""
            tools_ran = False

            with st.spinner("جارٍ تجهيز الإجابة..."):
                if uploaded_image is not None:
                    answer, model_used = analyze_image(
                        question,
                        uploaded_image,
                    )

                elif enable_web_search or enable_code:
                    (
                        answer,
                        sources,
                        model_used,
                        tools_ran,
                    ) = research_with_web(
                        question,
                        use_search=enable_web_search,
                        use_code=enable_code,
                    )

                else:
                    answer, model_used = call_chat(question)

                if not answer.strip():
                    answer = (
                        "لم يرجع النموذج إجابة نصية. "
                        "حاول إعادة صياغة السؤال."
                    )

                review_allowed = (
                    enable_review
                    and uploaded_image is None
                    and not enable_web_search
                    and not enable_code
                    and not tools_ran
                )

                if review_allowed:
                    with st.spinner("جارٍ مراجعة الإجابة..."):
                        answer = review_answer(
                            question,
                            answer,
                            model_used,
                        )
                        answer += (
                            FOOTER_MARK
                            + "🛡️ مراجعة آلية من النموذج نفسه من معرفته "
                            "المخزنة؛ ليست تحققًا بمصادر."
                        )

            st.session_state.messages.append({
                "role": "user",
                "content": saved_question,
            })

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })

            st.session_state.last_sources = sources
            st.session_state.last_model_used = model_used
            st.session_state.audit_results = []

            st.rerun()

        except Exception as error:
            st.error(
                "تعذر إكمال الطلب. تحقق من النموذج المختار "
                "ومفتاح API وحدود الاستخدام ودعم الأدوات."
            )

            with st.expander("تفاصيل الخطأ"):
                st.code(str(error))

            st.info(
                "إذا كان الخطأ متعلقًا بالبحث أو تشغيل الكود، "
                "جرّب تعطيل الأداة مؤقتًا. إذا كان الخطأ من النموذج، "
                "جرّب اختيار نموذج آخر من القائمة."
            )


import os
import io
import json
import base64
import re
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from groq import Groq

from reference_checker import (
    verify_reference,
    display_reference_result,
)


# =========================================================
# 1. الإعدادات
# =========================================================

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
]

TOOL_MODEL_PREFERENCES = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

SEARCH_MODELS = set(TOOL_MODEL_PREFERENCES)
CODE_MODELS = set(TOOL_MODEL_PREFERENCES)


# =========================================================
# 2. تعليمات المساعد
# =========================================================

SYSTEM_PROMPT = """
أنت مساعد ذكاء اصطناعي عام متعدد المجالات.

الأهداف:
الإجابة بدقة، وحل المشكلات، وشرح المعلومات، والبحث والتحقق،
والتعليم والبرمجة وتحليل الصور والبيانات.

قواعد الدقة:
1. لا تختلق حقائق أو أرقامًا أو دراسات أو مؤلفين أو روابط.
2. لا تدّع أنك استخدمت أداة لم تستخدمها فعلًا.
3. ميّز بين الحقيقة والاستنتاج والفرضية والرأي.
4. إذا لم تعرف، فقل ذلك بوضوح.
5. لا تخلط بين عدم العثور على دليل وبين إثبات خطأ الادعاء.
6. لا تعمم نتائج دراسة واحدة دون مبرر.
7. لا تخلط بين الارتباط والسببية.
8. لا توافق المستخدم لمجرد إرضائه.
9. صحح أخطاءك السابقة إذا اكتشفتها.
10. راجع الأرقام والوحدات والتواريخ والأسماء.

قواعد المصادر:
- اسم الدراسة أو وجود الرابط لا يثبت صحة الادعاء.
- لا تختلق عنوانًا أو اسم مؤلف أو سنة أو اسم مجلة.
- فرّق بين العثور على المصدر وفحص محتواه.
- فرّق بين صحة المعلومة وصحة المرجع الذي استُشهد به.
- لا تقل إن مرجعًا متحقق منه إلا إذا فُحصت بياناته فعلًا.
- لا تقل إن دراسة تدعم ادعاءً إلا إذا توفر دليل على ذلك.
- إذا تعذر التحقق، صرّح بذلك.
- عند وجود تعارض بين الأدلة، اذكره.
- فضّل المصادر الأصلية والرسمية والدراسات المحكمة.
- لا تعرض سلسلة التفكير الداخلية؛ قدم أسبابًا ونتائج قابلة للفحص.

في تدقيق الحقائق، استخدم الأحكام:
صحيح، صحيح جزئيًا، خاطئ، غير مدعوم، أو غير محسوم.

في البرمجة:
- لا تدّع تشغيل الكود دون تشغيله.
- حافظ على الوظائف الموجودة عند تعديل مشروع.
- لا تكشف مفاتيح API.
- وضّح ما لم يتم اختباره.

الأسلوب:
- أجب بلغة المستخدم.
- كن مباشرًا دون حشو.
- ابدأ بالنتيجة ثم الأدلة المهمة.
- صرّح بعدم اليقين بدل اختراع إجابة.
"""

RESEARCH_PROMPT = """
وضع البحث والتحقق:
استخدم أداة البحث المتاحة إن كانت مدعومة.
افحص الادعاءات الأساسية كلًا على حدة.
فضّل المصادر الأصلية والدراسات المحكمة.
اعرض الروابط الفعلية التي توفرها الأداة.
لا تختلق أسماء دراسات أو مؤلفين أو روابط.
لا تعتبر وجود الرابط دليلًا على صحة الادعاء.
إذا لم تفحص محتوى المصدر، فصرّح بذلك.
إذا لم تكف الأدلة، فقل إن الادعاء غير متحقق.
افصل بين صحة الفكرة وصحة تفاصيل المرجع.
"""

MODES = {
    "متوازن": "أجب بوضوح ودقة وباختصار مناسب.",
    "تحليل عميق": """
حلل المشكلة بعناية. افحص الافتراضات والبدائل والتناقضات
والقيود. اعرض النتائج والأسباب المهمة دون عرض سلسلة التفكير
الداخلية. لا تطل إذا كانت المسألة بسيطة.
""",
    "بحث وتحقق": RESEARCH_PROMPT,
    "معلّم خطوة بخطوة": """
اشرح الفكرة من الأساس إلى التطبيق. استخدم خطوات واضحة
وأمثلة عند الحاجة. لا تفترض أن المستخدم يعرف المصطلحات.
""",
}


# =========================================================
# 3. تصميم الواجهة
# =========================================================

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

    [data-testid="stFileUploader"] {
        color: #ffffff;
    }

    [data-testid="stExpander"] {
        border: 1px solid #414249;
        border-radius: 12px;
    }

    [data-testid="stChatMessage"] {
        border-radius: 14px;
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


# =========================================================
# 4. Groq
# =========================================================

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
        "تعذر جلب النماذج من Groq. تحقق من المفتاح "
        "والاتصال وحالة حسابك."
    )
    st.stop()


def available_preferences(preferences):
    return [m for m in preferences if m in available_models]


chat_options = available_preferences(CHAT_PREFERENCES)
vision_options = available_preferences(VISION_PREFERENCES)
tool_options = available_preferences(TOOL_MODEL_PREFERENCES)

search_options = [m for m in tool_options if m in SEARCH_MODELS]
code_options = [m for m in tool_options if m in CODE_MODELS]

if not chat_options:
    st.error(
        "لم يظهر نموذج دردشة من قائمة الاختيارات. "
        "راجع النماذج المتاحة في حساب Groq."
    )
    st.stop()


# =========================================================
# 5. حالة الجلسة
# =========================================================

DEFAULTS = {
    "messages": [],
    "last_sources": [],
    "last_model_used": "",
    "selected_chat_model": chat_options[0],
    "selected_mode": "متوازن",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.session_state.selected_chat_model not in chat_options:
    st.session_state.selected_chat_model = chat_options[0]

if st.session_state.selected_mode not in MODES:
    st.session_state.selected_mode = "متوازن"


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
                "content": item["content"],
            })

    while history and history[0]["role"] == "assistant":
        history.pop(0)

    return history


def make_messages(user_text, include_user=True):
    mode_prompt = MODES.get(
        st.session_state.selected_mode,
        MODES["متوازن"],
    )

    messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT + "\n\n" + mode_prompt,
    }]

    messages.extend(build_history())

    if include_user:
        messages.append({
            "role": "user",
            "content": user_text,
        })

    return messages


# =========================================================
# 6. أداة الاستجابة من Groq
# =========================================================

def completion_text(response):
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return ""


def call_completion(model, messages, tools=None, tool_choice=None):
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS,
    }

    if tools:
        kwargs["tools"] = tools

    if tool_choice:
        kwargs["tool_choice"] = tool_choice

    return client.chat.completions.create(**kwargs)


def call_chat(user_text, enable_code=False):
    model = st.session_state.selected_chat_model
    tools = None

    if enable_code:
        if model in CODE_MODELS:
            tools = [{"type": "code_interpreter"}]
        elif code_options:
            model = code_options[0]
            tools = [{"type": "code_interpreter"}]

    response = call_completion(
        model=model,
        messages=make_messages(user_text),
        tools=tools,
    )

    return completion_text(response), model


def extract_sources(response_data, answer=""):
    found = []

    def add(url, title=""):
        if isinstance(url, dict):
            url = url.get("url") or url.get("uri")

        if not isinstance(url, str):
            return

        url = url.strip().rstrip(".,;)]}")

        if re.match(r"^https?://", url, re.I):
            found.append({
                "title": str(title or url),
                "url": url,
            })

    def walk(value):
        if isinstance(value, dict):
            title = (
                value.get("title")
                or value.get("name")
                or value.get("source_title")
                or ""
            )

            for key in ("url", "uri", "href", "link"):
                if key in value:
                    add(value[key], title)

            for child in value.values():
                walk(child)

        elif isinstance(value, list):
            for child in value:
                walk(child)

        elif isinstance(value, str):
            for url in re.findall(
                r"https?://[^\s<>\"']+",
                value,
                flags=re.I,
            ):
                add(url)

    walk(response_data)
    walk(answer)

    unique = []
    seen = set()

    for source in found:
        if source["url"] not in seen:
            seen.add(source["url"])
            unique.append(source)

    return unique[:20]


def research_with_web(user_text, enable_code=False):
    if not search_options:
        raise RuntimeError(
            "البحث المدمج غير متاح ضمن النماذج التي ظهرت في حسابك."
        )

    selected = st.session_state.selected_chat_model
    model = selected if selected in SEARCH_MODELS else search_options[0]

    tools = [{"type": "browser_search"}]

    if enable_code and model in CODE_MODELS:
        tools.append({"type": "code_interpreter"})

    messages = make_messages(user_text)
    messages[0]["content"] += "\n\n" + RESEARCH_PROMPT

    response = call_completion(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="required",
    )

    answer = completion_text(response)

    try:
        response_data = response.model_dump()
    except Exception:
        response_data = {}

    sources = extract_sources(response_data, answer)

    if not answer.strip():
        answer = "لم يرجع النموذج إجابة نصية. حاول مرة أخرى."

    if not sources:
        answer += (
            "\n\n**تنبيه:** لم أستطع استخراج روابط مباشرة من "
            "بيانات الاستجابة. لا يعني ذلك أن الإجابة خاطئة، "
            "لكن التوثيق يحتاج إلى فحص إضافي."
        )

    return answer, sources, model


# =========================================================
# 7. تحليل الصور
# =========================================================

def image_to_data_url(image_file):
    image_file.seek(0)
    raw = image_file.read()

    if len(raw) > IMAGE_MAX_BYTES:
        raise ValueError("حجم الصورة يتجاوز 10 ميغابايت.")

    with Image.open(io.BytesIO(raw)) as original:
        image = original.convert("RGB")
        image.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))

        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=82,
            optimize=True,
        )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return "data:image/jpeg;base64," + encoded


def analyze_image(user_text, image_file):
    if not vision_options:
        raise RuntimeError("لم يظهر نموذج صور مناسب في قائمة النماذج.")

    model = vision_options[0]
    image_url = image_to_data_url(image_file)

    prompt = user_text.strip() or (
        "حلل الصورة بدقة. افصل بين ما يظهر بوضوح "
        "وما لا تستطيع التأكد منه."
    )

    messages = make_messages("", include_user=False)

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            },
        ],
    })

    response = call_completion(model, messages)
    return completion_text(response), model


# =========================================================
# 8. مراجعة الإجابة اختياريًا
# =========================================================

def review_answer(question, answer, model):
    prompt = """
راجع الإجابة بحثًا عن الأخطاء الواقعية والمنطقية والحسابية،
والتناقضات والمبالغة والادعاءات غير المدعومة.
صحح الأخطاء الواضحة، وصرّح بما لا تستطيع حسمه.
لا تختلق مصادر أو نتائج بحث.
لا تعرض سلسلة التفكير الداخلية.
أعد الإجابة النهائية المصححة فقط.
"""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + "\n\n" + prompt,
        },
        {
            "role": "user",
            "content": (
                "السؤال:\n" + question
                + "\n\nالإجابة المراد مراجعتها:\n" + answer
            ),
        },
    ]

    response = call_completion(model, messages)
    revised = completion_text(response)

    return revised.strip() if revised.strip() else answer


# =========================================================
# 9. زر النسخ
# =========================================================

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


# =========================================================
# 10. الشريط الجانبي
# =========================================================

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
        st.warning("لم يظهر نموذج صور مناسب.")

    st.divider()
    st.subheader("🧰 الأدوات")

    enable_web_search = st.toggle(
        "🔎 البحث في الإنترنت",
        value=False,
        disabled=not bool(search_options),
    )

    enable_code = st.toggle(
        "🧮 تشغيل الكود والحسابات",
        value=False,
        disabled=not bool(code_options),
    )

    enable_review = st.toggle(
        "🛡️ مراجعة إضافية للإجابة",
        value=False,
        help="قد تزيد استهلاك API، ولا تضمن خلو الإجابة من الأخطاء.",
    )

    # -----------------------------------------------
    # التحقق المستقل من المرجع - من الملف الخارجي
    # -----------------------------------------------

    st.divider()
    st.subheader("📚 التحقق من مرجع علمي")

    st.caption(
        "أدخل عنوان دراسة أو DOI أو رابط DOI. "
        "ستُفحص بيانات السجل في Crossref."
    )

    with st.form("reference_verification_form"):
        reference_query = st.text_input(
            "عنوان الدراسة أو DOI",
            placeholder="مثال: 10.xxxx/xxxxx",
        )

        verify_submitted = st.form_submit_button(
            "تحقق من المرجع",
            use_container_width=True,
        )

    if verify_submitted:
        try:
            with st.spinner("جارٍ البحث في سجل Crossref..."):
                reference_results = verify_reference(reference_query)

            if not reference_results:
                st.warning(
                    "لم تظهر نتائج في Crossref. هذا لا يثبت أن "
                    "المرجع مختلق؛ قد يكون خارج قاعدة البيانات."
                )
            else:
                for index, result in enumerate(reference_results):
                    with st.expander(
                        f"النتيجة {index + 1}: {result['title']}",
                        expanded=(index == 0),
                    ):
                        display_reference_result(result, st)

        except Exception as error:
            st.error(
                "تعذر التحقق من المرجع. قد تكون هناك مشكلة "
                "اتصال أو أن Crossref لم يجد DOI المطلوب."
            )
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
                st.success("تمت استعادة سجل المحادثة.")
                st.rerun()

            except Exception:
                st.error("تعذرت استعادة الملف. تأكد من سلامة JSON.")

    if st.button("مسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.last_model_used = ""
        st.rerun()


# =========================================================
# 11. واجهة المساعد
# =========================================================

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
                message["content"],
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
                f"- [{source['title']}]({source['url']})"
            )

if st.session_state.last_model_used:
    st.caption(
        "النموذج المستخدم في آخر إجابة: "
        + st.session_state.last_model_used
    )


# =========================================================
# 12. إدخال السؤال والصورة
# =========================================================

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


# =========================================================
# 13. تنفيذ الطلب
# =========================================================

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

            with st.spinner("جارٍ تجهيز الإجابة..."):
                if uploaded_image is not None:
                    answer, model_used = analyze_image(
                        question,
                        uploaded_image,
                    )

                elif enable_web_search:
                    answer, sources, model_used = research_with_web(
                        question,
                        enable_code=enable_code,
                    )

                else:
                    answer, model_used = call_chat(
                        question,
                        enable_code=enable_code,
                    )

                if not answer.strip():
                    answer = (
                        "لم يرجع النموذج إجابة نصية. "
                        "حاول إعادة صياغة السؤال."
                    )

                if enable_review and uploaded_image is None:
                    with st.spinner("جارٍ مراجعة الإجابة..."):
                        answer = review_answer(
                            question,
                            answer,
                            model_used,
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

            st.rerun()

        except Exception as error:
            st.error(
                "تعذر إكمال الطلب. تحقق من النموذج المختار "
                "ومفتاح API وحدود الاستخدام ودعم الأدوات."
            )

            with st.expander("تفاصيل الخطأ"):
                st.code(str(error))

            st.info(
                "جرّب تعطيل البحث وتشغيل الكود والمراجعة، "
                "ثم أعد السؤال."
            )

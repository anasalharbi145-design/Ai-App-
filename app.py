
import os
import io
import json
import base64
import html
import re

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from groq import Groq


# ==================================================
# 1. الإعدادات
# ==================================================

st.set_page_config(
    page_title="المساعد الذكي الشامل",
    page_icon="🤖",
    layout="wide",
)

MAX_TOKENS = 2048
HISTORY_LIMIT = 12
IMAGE_MAX_SIDE = 1024
IMAGE_MAX_BYTES = 10 * 1024 * 1024

CHAT_PREFERENCES = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
]

VISION_PREFERENCES = [
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-27b",
]

SEARCH_PREFERENCES = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

CODE_TOOL_MODELS = {
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
}

SEARCH_TOOL_MODELS = set(SEARCH_PREFERENCES)


SYSTEM_PROMPT = """
أنت مساعد ذكاء اصطناعي عام متعدد المجالات.

هدفك حل المشكلات والإجابة بدقة، وليس إنتاج إجابات تبدو مقنعة فقط.

المجالات:
العلوم، الرياضيات، البرمجة، التعليم، البحث، الكتابة،
الترجمة، التخطيط، تحليل البيانات والصور.

قواعد الدقة:
1. لا تختلق حقائق أو أرقامًا أو دراسات أو أسماء باحثين.
2. لا تختلق روابط أو مراجع أو نتائج بحث.
3. ميّز بين الحقيقة المثبتة والاستنتاج والفرضية والرأي.
4. إذا لم تعرف أو لم تكف الأدلة، فقل ذلك.
5. لا تصغ الأدلة الضعيفة كأنها يقينية.
6. تكرار الادعاء في مواقع متعددة ليس دليلًا مستقلًا بالضرورة.
7. لا تخلط بين الارتباط والسببية.
8. لا تعمم نتائج دراسة واحدة على جميع الحالات.
9. راجع الأرقام والوحدات والتواريخ والأسماء.
10. صحح افتراضات المستخدم الخاطئة بأدب ووضوح.
11. لا توافق المستخدم لمجرد إرضائه.
12. لا تدّع استخدام أداة لم تستخدمها فعلًا.

البحث والمصادر:
- استخدم نتائج البحث الفعلية عند توفر أداة البحث.
- فضّل المصادر الأصلية والرسمية والأبحاث المحكمة.
- تحقق من مدى ارتباط المصدر بالادعاء المحدد.
- لا تعتبر عنوان الصفحة وحده دليلًا على محتواها.
- لا تخترع أسماء الدراسات أو الروابط.
- فرّق بين مصدر عُرض في نتائج البحث ومصدر فُحص محتواه.
- إذا لم تكف النتائج، صرّح بأن التحقق غير مكتمل.
- لا تعتبر رابطًا ظهر في الإجابة إثباتًا لصحة الادعاء.
- عند تدقيق قائمة، افحص كل ادعاء على حدة.
- استخدم التصنيفات: صحيح، صحيح جزئيًا، خاطئ،
  غير مدعوم، أو لا تكفي الأدلة للحكم.

الرياضيات:
- تحقق من خطوات الحل والناتج والوحدات.
- وضّح الافتراضات المؤثرة.
- استخدم أدوات الحساب المتاحة عندما تفيد.
- لا تدّع تشغيل حساب برمجي دون تشغيله.

البرمجة:
- افحص الاستيرادات وأنواع البيانات وتدفق التنفيذ.
- حافظ على الوظائف الموجودة عند تعديل مشروع.
- لا تدّع اختبار الكود دون تشغيله.
- لا تكشف مفاتيح API أو الأسرار.
- ميّز بين الكود المقترح والكود الذي اختُبر.

الصور والبيانات:
- استنتج من المدخلات المتاحة فقط.
- وضّح عدم اليقين إذا كانت الصورة غير واضحة.
- لا تخترع تفاصيل غير موجودة.

الأسلوب:
- أجب بلغة المستخدم.
- كن مباشرًا وتجنب الحشو.
- أجب باختصار في الأسئلة البسيطة.
- نظّم الإجابات المعقدة إلى نتائج وأدلة وقيود.
- لا تعرض سلسلة التفكير الداخلية؛ قدم النتيجة
  والأسباب المفيدة للتحقق منها.

قبل إرسال الإجابة:
راجع الدقة، والتناقضات، والوحدات، والمصادر،
وحدود الأدلة. صحح الأخطاء الممكنة قبل الإرسال.
"""


RESEARCH_PROMPT = """
وضع البحث والتحقق:

استخدم أداة البحث المتاحة في هذا الطلب.
افحص الادعاءات بشكل منفصل.
ابحث عن المصادر الأصلية عندما يكون ذلك ممكنًا.
اعرض عنوان المصدر والرابط الفعلي إذا توفرا.
اشرح أي جزء من الادعاء يدعمه المصدر.
لا تخترع بيانات ببليوغرافية أو روابط.
لا تعتبر وجود الرابط دليلًا كافيًا على صحة الادعاء.
إذا لم تتمكن من فحص محتوى مصدر، فصرّح بذلك.
إذا تعارضت الأدلة، وضّح التعارض.
إذا لم تكف الأدلة، استخدم عبارة «غير متحقق».
لا تدّع أنك تحققت من جميع الادعاءات إذا لم يحدث ذلك.

نظّم الإجابة عند الحاجة إلى:
1. النتيجة المختصرة.
2. حكم كل ادعاء.
3. الأدلة والقيود.
4. المصادر التي أمكن تحديدها فعلًا.
"""


# ==================================================
# 2. التصميم
# ==================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #101114;
        color: #ffffff;
    }

    h1, h2, h3, p, label {
        color: #ffffff;
    }

    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #111111 !important;
        border-radius: 10px;
    }

    .stTextArea textarea::placeholder {
        color: #555555 !important;
    }

    .stButton button {
        border-radius: 10px;
        font-weight: 600;
    }

    [data-testid="stFileUploader"] {
        color: #ffffff;
    }

    [data-testid="stExpander"] {
        border: 1px solid #444444;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# 3. الاتصال بـ Groq
# ==================================================

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
        "أضفه في Secrets بإعدادات التطبيق."
    )
    st.stop()

client = Groq(api_key=GROQ_API_KEY)


# ==================================================
# 4. جلب النماذج واختيار المتاح
# ==================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_available_models():
    result = client.models.list()
    return sorted(model.id for model in result.data)


def preferred_available(preferences, available):
    return [
        model for model in preferences
        if model in available
    ]


try:
    available_models = get_available_models()
except Exception:
    st.error(
        "تعذر جلب قائمة النماذج من Groq. "
        "تحقق من مفتاح API والاتصال وحدود حسابك."
    )
    st.stop()


chat_options = preferred_available(
    CHAT_PREFERENCES,
    available_models,
)

vision_options = preferred_available(
    VISION_PREFERENCES,
    available_models,
)

search_options = preferred_available(
    SEARCH_PREFERENCES,
    available_models,
)

if not chat_options:
    st.error(
        "لم أجد نموذج دردشة من النماذج المدعومة في قائمة "
        "الاختيارات الحالية. راجع النماذج المتاحة في حساب Groq."
    )
    st.stop()


# ==================================================
# 5. ذاكرة الجلسة
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_search_used" not in st.session_state:
    st.session_state.last_search_used = False

if "selected_chat_model" not in st.session_state:
    st.session_state.selected_chat_model = chat_options[0]

if st.session_state.selected_chat_model not in chat_options:
    st.session_state.selected_chat_model = chat_options[0]


# ==================================================
# 6. زر نسخ الإجابة
# ==================================================

def render_copy_button(text, button_id):
    safe_text = json.dumps(
        text,
        ensure_ascii=False,
    ).replace("</", "<\\/")

    safe_id = str(button_id)

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
          <button id="copy-{safe_id}" type="button">
            📋 نسخ الإجابة
          </button>
          <span id="status-{safe_id}" role="status"></span>
          <script>
            const copyText = {safe_text};
            const button = document.getElementById("copy-{safe_id}");
            const status = document.getElementById("status-{safe_id}");

            button.addEventListener("click", async () => {{
              status.textContent = "";

              try {{
                if (!navigator.clipboard || !window.isSecureContext) {{
                  throw new Error("Clipboard unavailable");
                }}
                await navigator.clipboard.writeText(copyText);
                status.textContent = "تم النسخ ✓";
              }} catch (error) {{
                try {{
                  const area = document.createElement("textarea");
                  area.value = copyText;
                  area.setAttribute("readonly", "");
                  area.style.position = "fixed";
                  area.style.left = "-9999px";
                  document.body.appendChild(area);
                  area.focus();
                  area.select();
                  area.setSelectionRange(0, area.value.length);

                  const success = document.execCommand("copy");
                  document.body.removeChild(area);

                  if (!success) {{
                    throw new Error("Copy failed");
                  }}

                  status.textContent = "تم النسخ ✓";
                }} catch (fallbackError) {{
                  status.textContent = "تعذر النسخ التلقائي.";
                }}
              }}
            }});
          </script>
        </body>
        </html>
        """,
        height=48,
        scrolling=False,
    )


# ==================================================
# 7. سجل المحادثة
# ==================================================

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

    # لا تبدأ المحادثة برسالة مساعد يتيمة.
    while history and history[0]["role"] == "assistant":
        history.pop(0)

    return history


# ==================================================
# 8. معالجة الصور
# ==================================================

def image_to_data_url(image_file):
    image_file.seek(0)

    raw = image_file.read()

    if len(raw) > IMAGE_MAX_BYTES:
        raise ValueError(
            "حجم الصورة أكبر من الحد المسموح به (10 ميغابايت)."
        )

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


# ==================================================
# 9. استخراج المصادر والروابط
# ==================================================

def extract_sources(response_data, answer=""):
    """
    تستخرج الروابط والعناوين الموجودة فعلًا في بيانات الاستجابة.
    لا تثبت هذه الدالة أن محتوى الرابط يدعم الادعاء.
    """
    found = []

    def add(url, title=""):
        if isinstance(url, dict):
            url = url.get("url") or url.get("uri")

        if not isinstance(url, str):
            return

        url = url.strip().rstrip(".,;)]}")

        if not re.match(r"^https?://", url, re.I):
            return

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

    for item in found:
        if item["url"] not in seen:
            seen.add(item["url"])
            unique.append(item)

    return unique[:20]


# ==================================================
# 10. إنشاء طلب للنموذج
# ==================================================

def make_messages(user_text, system_prompt=SYSTEM_PROMPT):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    messages.extend(build_history())

    messages.append({
        "role": "user",
        "content": user_text,
    })

    return messages


def completion_text(response):
    return response.choices[0].message.content or ""


# ==================================================
# 11. الدردشة العادية وأداة تشغيل الكود
# ==================================================

def call_chat(user_text, enable_code=False):
    model = st.session_state.selected_chat_model

    messages = make_messages(user_text)
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS,
    }

    # أداة تشغيل الكود متاحة مع نماذج GPT-OSS المحددة فقط.
    if enable_code and model in CODE_TOOL_MODELS:
        kwargs["tools"] = [
            {"type": "code_interpreter"}
        ]
        kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)
    return completion_text(response)


# ==================================================
# 12. البحث في الإنترنت
# ==================================================

def research_with_web(user_text, enable_code=False):
    if not search_options:
        raise RuntimeError(
            "لا يوجد نموذج يدعم البحث المدمج في قائمة حسابك."
        )

    # البحث يتطلب نموذجًا يدعم browser_search،
    # حتى لو كان نموذج الدردشة المختار مختلفًا.
    model = (
        st.session_state.selected_chat_model
        if st.session_state.selected_chat_model in SEARCH_TOOL_MODELS
        else search_options[0]
    )

    system_prompt = SYSTEM_PROMPT + "\n\n" + RESEARCH_PROMPT
    messages = make_messages(user_text, system_prompt)

    tools = [{"type": "browser_search"}]

    if enable_code and model in CODE_TOOL_MODELS:
        tools.append({"type": "code_interpreter"})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="required",
        temperature=0.2,
        max_tokens=MAX_TOKENS,
    )

    answer = completion_text(response)

    try:
        response_data = response.model_dump()
    except Exception:
        response_data = {}

    sources = extract_sources(response_data, answer)

    if not answer.strip():
        answer = "لم يرجع البحث إجابة نصية. حاول صياغة السؤال بوضوح."

    if not sources:
        answer += (
            "\n\n**تنبيه توثيق:** لم أستطع استخراج روابط مباشرة "
            "من بيانات الاستجابة. قد يحتوي النص على إحالات داخلية "
            "للبحث، لكنني لا أملك هنا روابط مستقلة كافية لعرضها. "
            "هذا لا يعني أن الإجابة خاطئة، ولا يعني أن مصادرها قد "
            "تم التحقق منها بالكامل."
        )

    return answer, sources, model


# ==================================================
# 13. تحليل الصور
# ==================================================

def analyze_image(user_text, image_file):
    if not vision_options:
        raise RuntimeError(
            "لم أجد نموذج رؤية مدعومًا ضمن النماذج المتاحة."
        )

    model = (
        VISION_PREFERENCES[0]
        if VISION_PREFERENCES[0] in vision_options
        else vision_options[0]
    )

    image_url = image_to_data_url(image_file)

    prompt = user_text or (
        "حلل الصورة بدقة. افصل بين ما يظهر بوضوح "
        "وما لا تستطيع التأكد منه."
    )

    messages = make_messages(prompt)

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

    # إزالة آخر رسالة نصية أضافتها make_messages؛
    # فالرسالة الأخيرة الآن تحتوي السؤال والصورة معًا.
    messages.pop(-2)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=MAX_TOKENS,
    )

    return completion_text(response), model


# ==================================================
# 14. الشريط الجانبي
# ==================================================

with st.sidebar:
    st.header("⚙️ الإعدادات")

    selected_chat = st.selectbox(
        "نموذج الدردشة",
        options=chat_options,
        index=chat_options.index(
            st.session_state.selected_chat_model
        ),
    )

    st.session_state.selected_chat_model = selected_chat

    if vision_options:
        st.caption(
            "نموذج الصور: "
            + (
                VISION_PREFERENCES[0]
                if VISION_PREFERENCES[0] in vision_options
                else vision_options[0]
            )
        )
    else:
        st.warning("لا يوجد نموذج صور مناسب ظاهر في حسابك.")

    st.caption(f"النماذج المتاحة في القائمة: {len(available_models)}")

    st.divider()
    st.subheader("🧰 الأدوات")

    enable_web_search = st.toggle(
        "🔎 البحث في الإنترنت",
        value=False,
        disabled=not bool(search_options),
    )

    if not search_options:
        st.warning("البحث المدمج غير متاح ضمن النماذج التي ظهرت.")

    code_supported = (
        selected_chat in CODE_TOOL_MODELS
        or bool(search_options)
    )

    enable_code = st.toggle(
        "🧮 السماح بأداة تشغيل الكود",
        value=False,
        disabled=not code_supported,
        help="تستخدم الأداة في الحسابات وتحليل البيانات عند دعم النموذج لها.",
    )

    st.caption(
        "الأدوات لا تضمن صحة النتائج؛ يجب مراجعة الأدلة "
        "ومخرجات الحساب عند الحاجة."
    )

    st.divider()
    st.subheader("💾 المحادثات")

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
                st.session_state.last_search_used = False

                st.success("تمت استعادة سجل المحادثة.")
                st.rerun()

            except Exception:
                st.error("تعذرت استعادة الملف. تأكد من سلامة JSON.")

    if st.button("مسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.last_search_used = False
        st.rerun()


# ==================================================
# 15. واجهة المساعد
# ==================================================

st.title("🤖 المساعد الذكي الشامل")

st.caption(
    "دردشة • بحث • تحليل صور • سجل محادثات • أدوات اختيارية"
)

if not st.session_state.messages:
    st.info(
        "اكتب سؤالك. فعّل البحث عندما تحتاج معلومات حديثة "
        "أو تريد التحقق من الادعاءات والمصادر."
    )


# عرض السجل
for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_copy_button(
                message["content"],
                f"history_{index}",
            )


# عرض مصادر الاستجابة الأخيرة
if st.session_state.last_sources:
    with st.expander("🔗 الروابط التي ظهرت في الاستجابة"):
        st.warning(
            "ظهور الرابط لا يثبت بمفرده أن محتواه يدعم الادعاء. "
            "افتح المصدر وراجع الدليل المرتبط بالمعلومة."
        )

        for source in st.session_state.last_sources:
            st.markdown(
                f"- [{source['title']}]({source['url']})"
            )


# ==================================================
# 16. إدخال السؤال والصورة
# ==================================================

with st.form("chat_form", clear_on_submit=True):
    user_text = st.text_area(
        "اكتب سؤالك",
        placeholder="اسأل عن أي موضوع...",
        height=120,
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


# ==================================================
# 17. معالجة الطلب
# ==================================================

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
            with st.spinner("جارٍ تجهيز الإجابة..."):
                sources = []
                used_search = False
                model_used = ""

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
                    used_search = True

                else:
                    answer = call_chat(
                        question,
                        enable_code=enable_code,
                    )
                    model_used = st.session_state.selected_chat_model

            if not answer.strip():
                answer = "لم يرجع النموذج إجابة نصية. حاول مرة أخرى."

            st.session_state.messages.append({
                "role": "user",
                "content": saved_question,
            })

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })

            st.session_state.last_sources = sources
            st.session_state.last_search_used = used_search

            st.session_state.last_model_used = model_used

            st.rerun()

        except Exception as e:
            st.error(
                "تعذر إكمال الطلب. تحقق من النموذج المختار "
                "ومفتاح API وحدود الاستخدام."
            )

            with st.expander("تفاصيل الخطأ"):
                st.code(str(e))

            st.info(
                "إذا كان الخطأ متعلقًا بأداة البحث أو تشغيل الكود، "
                "جرّب تعطيل الأداة ثم أعد الطلب."
            )

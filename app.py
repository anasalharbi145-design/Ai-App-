
import os
import io
import json
import base64
import re

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from groq import Groq


# =========================================================
# 1. إعدادات التطبيق
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
# 2. تعليمات الذكاء الاصطناعي
# =========================================================

SYSTEM_PROMPT = """
أنت مساعد ذكاء اصطناعي عام، دقيق، عملي، ومتعدد المجالات.
مهمتك مساعدة المستخدم على الفهم وحل المشكلات، لا مجرد إنتاج
إجابة مقنعة لغويًا.

المجالات:
العلوم، الرياضيات، البرمجة، البحث، التعليم، الكتابة، الترجمة،
تحليل الصور، تحليل البيانات، التخطيط، والتفكير النقدي.

أولًا: الدقة والنزاهة
- لا تختلق حقائق أو أرقامًا أو دراسات أو أسماء أشخاص أو مراجع.
- لا تدّع أنك بحثت أو اختبرت كودًا أو نفذت عملية لم تنفذها.
- ميّز بين المعلومة المؤكدة والاستنتاج والفرضية والتوقع والرأي.
- إذا كانت الأدلة غير كافية، قل ذلك بوضوح.
- إذا كان السؤال يعتمد على معلومات حديثة، فابحث إن توفرت الأداة.
- إذا لم تستخدم البحث، فلا توحِ بأنك تحققت من المعلومات حديثًا.
- لا تعتبر تكرار المعلومة في مواقع كثيرة دليلًا مستقلًا.
- لا تخلط بين الارتباط والسببية.
- لا تعمم نتائج دراسة واحدة دون مبرر.
- لا توافق المستخدم لمجرد إرضائه؛ صحح الأخطاء باحترام.
- إذا اكتشفت خطأ في إجابتك السابقة، اعترف به وصححه.
- لا تخلط بين عدم العثور على دليل وبين ثبوت أن الادعاء خاطئ.

ثانيًا: البحث والمصادر
عند استخدام البحث:
- فضّل المصادر الأصلية والرسمية والدراسات المحكمة.
- ابحث عن الدليل الذي يدعم الادعاء المحدد، لا العنوان فقط.
- ميّز بين صفحة ظهرت في النتائج وصفحة فُحص محتواها.
- لا تخترع روابط أو عناوين دراسات أو أسماء مؤلفين.
- احتفظ بالقيود والتعارضات المهمة بين المصادر.
- اذكر تاريخ المعلومة إذا كان مهمًا.
- لا تقل إن كل ادعاء تحقق بالكامل إذا لم يحدث ذلك.
- إذا لم تتوفر مصادر كافية، قل: لم أتمكن من التحقق الكافي.
- لا تعتبر وجود رابط وحده إثباتًا لصحة المعلومة.

ثالثًا: التفكير وحل المشكلات
- حدد المطلوب والقيود قبل الحل.
- استخدم الطريقة المناسبة للمسألة بدل تطبيق قالب واحد على كل شيء.
- افحص الافتراضات والوحدات والأرقام والحالات الاستثنائية.
- قارن بين التفسيرات المعقولة عندما يكون السؤال ملتبسًا.
- أعط النتيجة أولًا، ثم الأسباب والأدلة المفيدة.
- لا تعرض سلسلة التفكير الداخلية؛ قدم ملخصًا واضحًا للأسباب
  والخطوات التي يستطيع المستخدم التحقق منها.
- لا تتظاهر باليقين حين تكون المسألة غير محسومة.

رابعًا: الرياضيات والعلوم
- اشرح الرموز والقواعد عند حاجة المستخدم إليها.
- تحقق من الناتج والوحدات والإشارات والافتراضات.
- استخدم أداة تشغيل الكود المتاحة عند فائدتها.
- ميّز بين القياس التجريبي والتفسير النظري والفرضية.
- عند تدقيق الحقائق، صنفها إلى:
  صحيح، صحيح جزئيًا، خاطئ، غير مدعوم، أو غير محسوم.
- لا تستنتج دقة رقم من عدد منازل عشرية فقط.

خامسًا: البرمجة
- حافظ على الوظائف الموجودة عندما تعدل مشروعًا قائمًا.
- افحص الاستيرادات وتوافق المكتبات وأنواع البيانات.
- وضح المتطلبات والإعدادات اللازمة للتشغيل.
- لا تدّع اختبار الكود إلا إذا شُغّل فعلًا.
- لا تكشف مفاتيح API أو كلمات المرور.
- لا تضع الأسرار في الواجهة الأمامية أو في ملفات عامة.
- اذكر بوضوح أي جزء لم يتم اختباره.

سادسًا: الصور والبيانات
- اعتمد على ما يظهر فعلًا في الصورة أو البيانات.
- لا تخترع تفاصيل غير واضحة.
- اشرح القيود إذا كانت الدقة أو جودة المدخلات ضعيفة.
- لا تستنتج بيانات شخصية حساسة من صورة دون مبرر.

سابعًا: أسلوب التواصل
- استخدم لغة المستخدم.
- كن مباشرًا، واضحًا، ومنظمًا.
- لا تكثر من المقدمات والمديح والحشو.
- الأسئلة البسيطة تستحق إجابات قصيرة.
- الأسئلة المعقدة تستحق شرحًا كافيًا دون تكرار.
- اسأل سؤالًا توضيحيًا واحدًا إذا كان الغموض يمنع الحل.
- لا تعد المستخدم بنتيجة لا تستطيع ضمانها.

قبل إرسال الإجابة:
راجع الأرقام والتناقضات والاستنتاجات والمصادر والقيود.
صحح الأخطاء الواضحة، وصرّح بما بقي غير مؤكد.
"""


RESEARCH_PROMPT = """
وضع البحث والتحقق:
استخدم أداة البحث المتاحة في هذا الطلب.
ابحث عن معلومات تدعم الادعاءات الأساسية بشكل مباشر.
فضّل المصادر الأصلية والرسمية والأبحاث العلمية.
اعرض الروابط الفعلية التي توفرها نتائج البحث عندما تتوفر.
اشرح ما الذي يدعمه كل مصدر، وما الذي لا يدعمه.
لا تختلق اقتباسات أو روابط أو بيانات ببليوغرافية.
إذا لم تتمكن من التحقق من محتوى مصدر، فصرّح بذلك.
إذا تعارضت المصادر، وضّح التعارض ولا تخفِه.
إذا لم تكف الأدلة، صنّف الادعاء بأنه غير متحقق.
لا تدّع أن وجود المصادر يثبت صحة كل جملة.

عند تدقيق عدة ادعاءات، افحص كل ادعاء على حدة.
نظّم النتيجة إلى:
1. الحكم المختصر.
2. الأدلة لكل ادعاء.
3. القيود ومواطن عدم اليقين.
4. المصادر والروابط المتاحة فعلًا.
"""


MODES = {
    "متوازن": """
أجب بأفضل توازن بين الدقة والوضوح والاختصار.
ابدأ بالإجابة المباشرة ثم أضف التفاصيل الضرورية.
""",
    "تحليل عميق": """
حلل المشكلة بعناية.
حدد الافتراضات والقيود والبدائل والتفسيرات المحتملة.
افحص الاتساق والأرقام والأدلة قبل عرض النتيجة.
قدم ملخصًا للأسباب المهمة دون عرض سلسلة التفكير الداخلية.
لا تطل إذا كانت المسألة بسيطة.
""",
    "بحث وتحقق": RESEARCH_PROMPT,
    "معلّم خطوة بخطوة": """
تصرف كمعلم صبور.
اشرح الفكرة من الأساس إلى التطبيق.
قسّم المسائل الصعبة إلى خطوات واضحة، واستخدم مثالًا عند فائدته.
لا تفترض أن المستخدم يعرف المصطلحات المتقدمة.
في نهاية الشرح، قدم سؤالًا تدريبيًا قصيرًا إذا كان مناسبًا.
""",
}


# =========================================================
# 3. تنسيق الواجهة
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #101114;
        color: #f5f5f5;
    }

    [data-testid="stHeader"] {
        background: rgba(16, 17, 20, 0.96);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 900px;
        padding: 1rem 1rem 2rem;
    }

    h1, h2, h3, p, label {
        color: #f5f5f5;
    }

    .stTextArea textarea {
        background: #202126 !important;
        color: #ffffff !important;
        border: 1px solid #45464d !important;
        border-radius: 14px !important;
        font-size: 16px !important;
    }

    .stTextArea textarea::placeholder {
        color: #b6b6bd !important;
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

        [data-testid="stChatMessage"] {
            padding: 0.6rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 4. الاتصال الآمن بـ Groq
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


# =========================================================
# 5. اكتشاف النماذج المتاحة
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_available_models():
    result = client.models.list()
    return sorted(model.id for model in result.data)


try:
    available_models = get_available_models()
except Exception:
    st.error(
        "تعذر جلب النماذج من Groq. تحقق من مفتاح API "
        "والاتصال وحالة حسابك."
    )
    st.stop()


def available_preferences(preferences):
    return [m for m in preferences if m in available_models]


chat_options = available_preferences(CHAT_PREFERENCES)
vision_options = available_preferences(VISION_PREFERENCES)
tool_model_options = available_preferences(TOOL_MODEL_PREFERENCES)

search_options = [
    m for m in tool_model_options if m in SEARCH_MODELS
]
code_options = [
    m for m in tool_model_options if m in CODE_MODELS
]

if not chat_options:
    st.error(
        "لم يظهر أي نموذج دردشة من قائمة الاختيارات. "
        "راجع النماذج المتاحة في حساب Groq وحدّث قائمة "
        "CHAT_PREFERENCES داخل الملف عند الحاجة."
    )
    st.stop()


# =========================================================
# 6. حالة الجلسة
# =========================================================

DEFAULTS = {
    "messages": [],
    "last_sources": [],
    "selected_chat_model": chat_options[0],
    "selected_mode": "متوازن",
    "last_model_used": "",
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


def build_system_prompt():
    mode = st.session_state.selected_mode
    mode_prompt = MODES.get(mode, MODES["متوازن"])

    return SYSTEM_PROMPT + "\n\n" + mode_prompt


def make_messages(user_text, include_user=True):
    messages = [{
        "role": "system",
        "content": build_system_prompt(),
    }]

    messages.extend(build_history())

    if include_user:
        messages.append({
            "role": "user",
            "content": user_text,
        })

    return messages


# =========================================================
# 7. أدوات مساعدة
# =========================================================

def completion_text(response):
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return ""


def extract_sources(response_data, answer=""):
    """
    يستخرج الروابط الفعلية الموجودة في نص الإجابة
    أو بيانات الاستجابة. لا يثبت أن الرابط يدعم الادعاء.
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
            urls = re.findall(
                r"https?://[^\s<>\"']+",
                value,
                flags=re.I,
            )
            for url in urls:
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


def image_to_data_url(image_file):
    image_file.seek(0)
    raw = image_file.read()

    if len(raw) > IMAGE_MAX_BYTES:
        raise ValueError(
            "حجم الصورة يتجاوز 10 ميغابايت. "
            "اختر صورة أصغر."
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


def call_completion(model, messages, tools=None, tool_choice=None):
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_completion_tokens": MAX_TOKENS,
    }

    if tools:
        kwargs["tools"] = tools

    if tool_choice:
        kwargs["tool_choice"] = tool_choice

    return client.chat.completions.create(**kwargs)


# =========================================================
# 8. الدردشة
# =========================================================

def call_chat(user_text, enable_code=False):
    selected_model = st.session_state.selected_chat_model
    model = selected_model
    tools = None

    if enable_code:
        if selected_model in CODE_MODELS:
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


# =========================================================
# 9. البحث في الإنترنت
# =========================================================

def research_with_web(user_text, enable_code=False):
    if not search_options:
        raise RuntimeError(
            "البحث المدمج غير متاح ضمن النماذج الظاهرة في حسابك."
        )

    selected_model = st.session_state.selected_chat_model

    if selected_model in SEARCH_MODELS:
        model = selected_model
    else:
        model = search_options[0]

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
        answer = "لم يرجع البحث إجابة نصية. حاول مرة أخرى."

    if not sources:
        answer += (
            "\n\n**تنبيه توثيق:** لم أستطع استخراج روابط مباشرة "
            "من الاستجابة. لا يعني ذلك أن الإجابة خاطئة، "
            "لكنه يعني أن توثيقها بالروابط لم يكتمل."
        )

    return answer, sources, model


# =========================================================
# 10. تحليل الصور
# =========================================================

def analyze_image(user_text, image_file):
    if not vision_options:
        raise RuntimeError(
            "لم يظهر نموذج رؤية مناسب ضمن النماذج المتاحة."
        )

    model = vision_options[0]
    image_url = image_to_data_url(image_file)

    prompt = user_text.strip() or (
        "حلل الصورة بدقة، وافصل بين ما يظهر بوضوح "
        "وما لا تستطيع التأكد منه."
    )

    messages = make_messages("", include_user=False)

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

    response = call_completion(
        model=model,
        messages=messages,
    )

    return completion_text(response), model


# =========================================================
# 11. مراجعة الإجابة اختياريًا
# =========================================================

def review_answer(question, answer, model):
    """
    مراجعة إضافية اختيارية. ليست ضمانًا للصحة،
    ولا تغني عن التحقق من المصادر الأصلية.
    """
    review_prompt = """
راجع الإجابة التالية مراجعة نقدية.
ابحث عن الأخطاء الواقعية والمنطقية والحسابية،
والتناقضات والمبالغة والادعاءات غير المدعومة.
إذا وجدت خطأ واضحًا، صحح الإجابة.
إذا لم تستطع حسم ادعاء، وضّح عدم اليقين.
لا تخترع مصادر أو نتائج بحث.
لا تعرض سلسلة التفكير الداخلية.
أعد الإجابة النهائية المصححة فقط، مع الاحتفاظ
بالمعلومات المفيدة وعدم إطالة النص دون حاجة.
"""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + "\n\n" + review_prompt,
        },
        {
            "role": "user",
            "content": (
                "السؤال الأصلي:\n"
                + question
                + "\n\nالإجابة المراد مراجعتها:\n"
                + answer
            ),
        },
    ]

    response = call_completion(
        model=model,
        messages=messages,
    )

    revised = completion_text(response)
    return revised.strip() if revised.strip() else answer


# =========================================================
# 12. زر نسخ الإجابة
# =========================================================

def render_copy_button(text, button_id):
    safe_text = json.dumps(
        text,
        ensure_ascii=False,
    ).replace("</", "<\\/")

    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(button_id))

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


# =========================================================
# 13. الشريط الجانبي
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

    st.caption(
        "النماذج التي تظهر هنا متاحة في قائمة حسابك، "
        "لكن توفرها لا يضمن دعم كل أداة."
    )

    if vision_options:
        st.caption("نموذج الصور: " + vision_options[0])
    else:
        st.warning("لم يظهر نموذج صور متاح في قائمة الاختيارات.")

    st.divider()
    st.subheader("🧰 الأدوات")

    enable_web_search = st.toggle(
        "🔎 البحث في الإنترنت",
        value=False,
        disabled=not bool(search_options),
        help="البحث يستخدم نموذجًا يدعم الأداة عند الحاجة.",
    )

    enable_code = st.toggle(
        "🧮 تشغيل الكود والحسابات",
        value=False,
        disabled=not bool(code_options),
        help="يسمح للنموذج باستخدام أداة تشغيل الكود المدعومة.",
    )

    enable_review = st.toggle(
        "🛡️ مراجعة إضافية للإجابة",
        value=False,
        help=(
            "يرسل الإجابة لمراجعة ثانية؛ قد يزيد استهلاك API "
            "ولا يضمن خلو الإجابة من الأخطاء."
        ),
    )

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
                st.error(
                    "تعذرت استعادة الملف. تأكد من سلامة JSON."
                )

    if st.button("مسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()


# =========================================================
# 14. واجهة المساعد
# =========================================================

st.title("🤖 المساعد الذكي الشامل")

st.caption(
    "دردشة • بحث • تحليل صور • تعليم • برمجة • مراجعة اختيارية"
)

if not st.session_state.messages:
    st.info(
        "اكتب سؤالك. اختر وضع الإجابة من الإعدادات، "
        "وفعّل البحث عندما تحتاج معلومات حديثة أو توثيقًا."
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
            "ظهور الرابط لا يثبت بمفرده أن محتواه يدعم الادعاء. "
            "راجع المصدر نفسه قبل اعتماد المعلومة."
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
# 15. إدخال السؤال والصورة
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
# 16. تنفيذ الطلب
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
            used_search = False

            with st.spinner("جارٍ تحليل الطلب وتجهيز الإجابة..."):
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
                    answer, model_used = call_chat(
                        question,
                        enable_code=enable_code,
                    )

                if not answer.strip():
                    answer = (
                        "لم يرجع النموذج إجابة نصية. "
                        "جرّب إعادة صياغة السؤال."
                    )

                if enable_review and uploaded_image is None:
                    with st.spinner("تتم مراجعة الإجابة..."):
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
                "تعذر إكمال الطلب. قد يكون السبب نموذجًا غير متاح، "
                "أو أداة غير مدعومة، أو مشكلة في مفتاح API "
                "أو حدود الاستخدام."
            )

            with st.expander("تفاصيل الخطأ"):
                st.code(str(error))

            st.info(
                "جرّب تعطيل أدوات البحث وتشغيل الكود والمراجعة "
                "ثم أعد السؤال. لا تشارك تفاصيل الخطأ إذا كانت "
                "تحتوي على معلومات سرية."
            )

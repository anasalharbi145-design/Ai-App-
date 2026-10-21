import os
import io
import json
import base64

import streamlit as st
from PIL import Image
from groq import Groq


# =====================================
# 1. الإعدادات
# =====================================

st.set_page_config(
    page_title="مساعد الذكاء الاصطناعي",
    page_icon="🤖",
    layout="wide"
)

CHAT_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.8-27b"

MAX_TOKENS = 512
HISTORY_LIMIT = 12

SYSTEM_PROMPT = """
أنت مساعد ذكاء اصطناعي عربي دقيق وقوي في التحليل والرياضيات
والعلوم والبرمجة والكتابة والتعليم.

قواعد الدقة:
- لا تختلق حقائق أو مصادر أو روابط.
- لا تقل إنك بحثت في الإنترنت إلا إذا استُخدمت أداة البحث فعلًا.
- عند البحث، استند إلى نتائج البحث المتاحة، واذكر المصادر التي
  يوفرها النظام دون اختراع روابط.
- فرّق بين المعلومة المؤكدة والاستنتاج والفرضية.
- إذا لم تكن متأكدًا، قل ذلك بوضوح.
- تحقق من الحسابات والاستنتاجات قبل الإجابة.
- إذا كان السؤال يعتمد على معلومات حديثة، فابحث عند توفر أداة البحث.
- لا توافق المستخدم لمجرد إرضائه؛ صحح الخطأ بأدب.
- أجب بالعربية الواضحة وباختصار مناسب، إلا إذا طُلب التفصيل.
- لا تعرض التفكير الداخلي الخاص؛ قدم النتيجة والأسباب المفيدة فقط.
"""


# =====================================
# 2. التصميم
# =====================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #101114;
        color: #ffffff;
    }

    h1, h2, h3, p, label, span {
        color: #ffffff;
    }

    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 10px;
    }

    .stTextArea textarea::placeholder {
        color: #555555 !important;
    }

    .stButton button {
        border-radius: 10px;
        font-weight: bold;
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
    unsafe_allow_html=True
)


# =====================================
# 3. مفتاح Groq
# =====================================

def get_secret(name):
    value = os.getenv(name)

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
        "أضفه في Secrets داخل إعدادات التطبيق."
    )
    st.stop()

client = Groq(api_key=GROQ_API_KEY)


# =====================================
# 4. فحص النماذج
# =====================================

@st.cache_data(ttl=300)
def get_models():
    result = client.models.list()
    return sorted(model.id for model in result.data)


try:
    available_models = get_models()
except Exception as e:
    st.error(f"تعذر جلب النماذج: {e}")
    st.stop()


chat_available = CHAT_MODEL in available_models
vision_available = VISION_MODEL in available_models


# =====================================
# 5. ذاكرة المحادثة داخل الجلسة
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# =====================================
# 6. حفظ واستعادة سجل المحادثة
# =====================================

with st.sidebar:
    st.header("⚙️ الإعدادات")

    st.caption(f"الدردشة: {CHAT_MODEL}")
    st.caption(f"الصور: {VISION_MODEL}")

    if not chat_available:
        st.warning(
            "نموذج الدردشة غير ظاهر في قائمة حسابك في Groq."
        )

    if not vision_available:
        st.warning(
            "نموذج الصور غير ظاهر في قائمة حسابك في Groq."
        )

    st.divider()
    st.subheader("💾 ذاكرة المحادثة")

    history_json = json.dumps(
        st.session_state.messages,
        ensure_ascii=False,
        indent=2
    )

    st.download_button(
        "حفظ المحادثة كملف",
        data=history_json,
        file_name="chat_history.json",
        mime="application/json",
        use_container_width=True
    )

    uploaded_history = st.file_uploader(
        "استعادة محادثة محفوظة",
        type=["json"],
        key="history_import"
    )

    if st.button(
        "استعادة الملف",
        use_container_width=True
    ):
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
                        and item.get("role") in ["user", "assistant"]
                        and isinstance(item.get("content"), str)
                    ):
                        valid_messages.append({
                            "role": item["role"],
                            "content": item["content"]
                        })

                st.session_state.messages = valid_messages
                st.session_state.last_sources = []

                st.success("تمت استعادة سجل المحادثة.")
                st.rerun()

            except Exception as e:
                st.error(f"تعذرت استعادة الملف: {e}")

    if st.button(
        "مسح المحادثة الحالية",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()

    st.divider()

    enable_web_search = st.toggle(
        "🔎 البحث في الإنترنت",
        value=False
    )

    st.caption(
        "البحث يستخدم نموذج GPT-OSS 120B وأداة البحث المدمجة."
    )


# =====================================
# 7. تحويل الصورة
# =====================================

def image_to_data_url(image_file, max_side=768):
    image_file.seek(0)

    image = Image.open(image_file).convert("RGB")
    image.thumbnail((max_side, max_side))

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=70,
        optimize=True
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


# =====================================
# 8. تجهيز تاريخ المحادثة للنموذج
# =====================================

def build_history():
    history = []

    for item in st.session_state.messages[-HISTORY_LIMIT:]:
        if item["role"] in ["user", "assistant"]:
            history.append({
                "role": item["role"],
                "content": item["content"]
            })

    return history


# =====================================
# 9. استخراج الروابط إن أعادها API
# =====================================

def extract_urls(obj):
    found = []

    def walk(value):
        if isinstance(value, dict):
            url = value.get("url")
            title = value.get("title") or value.get("name")

            if (
                isinstance(url, str)
                and url.startswith(("https://", "http://"))
            ):
                found.append({
                    "title": str(title or url),
                    "url": url
                })

            for child in value.values():
                walk(child)

        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(obj)

    unique = []
    seen = set()

    for source in found:
        if source["url"] not in seen:
            seen.add(source["url"])
            unique.append(source)

    return unique


# =====================================
# 10. الدردشة العادية
# =====================================

def call_chat(user_text):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    messages.extend(build_history())

    messages.append({
        "role": "user",
        "content": user_text
    })

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=MAX_TOKENS
    )

    return response.choices[0].message.content or ""


# =====================================
# 11. البحث في الإنترنت
# =====================================

def research_with_web(user_text):
    if not chat_available:
        raise RuntimeError(
            "نموذج GPT-OSS 120B غير متاح في حسابك."
        )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
            + """
عند استخدام البحث:
- ابحث عن المعلومات ذات الصلة بالسؤال.
- أعط الأولوية للمصادر الأصلية والموثوقة والحديثة.
- لا تعتمد على نتيجة بحث لا تدعم الادعاء.
- اذكر تاريخ المعلومات عندما يكون مهمًا.
- لا تخترع روابط أو مراجع.
"""
        }
    ]

    messages.extend(build_history())

    messages.append({
        "role": "user",
        "content": user_text
    })

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=[
            {"type": "browser_search"}
        ],
        tool_choice="required",
        temperature=0.2,
        max_tokens=MAX_TOKENS
    )

    message = response.choices[0].message

    # نحاول التقاط روابط المصادر إن أتاحها الرد
    try:
        response_data = response.model_dump()
    except Exception:
        response_data = {}

    sources = extract_urls(response_data)

    return message.content or "", sources


# =====================================
# 12. تحليل الصور
# =====================================

def analyze_image(user_text, image_file):
    if not vision_available:
        raise RuntimeError(
            f"نموذج الصور {VISION_MODEL} غير متاح في حسابك."
        )

    image_url = image_to_data_url(image_file)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    messages.extend(build_history())

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    user_text
                    or "حلل الصورة واشرح محتواها بدقة. "
                       "ميّز بين ما تراه وما لا تستطيع التأكد منه."
                )
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }
        ]
    })

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=MAX_TOKENS
    )

    return response.choices[0].message.content or ""


# =====================================
# 13. واجهة المساعد
# =====================================

st.title("🤖 مساعد الذكاء الاصطناعي")

st.caption(
    "دردشة ذكية • تحليل الصور • بحث في الإنترنت • سجل محادثات"
)

if not st.session_state.messages:
    st.info(
        "ابدأ بسؤالك. فعّل البحث إذا احتجت معلومات حديثة ومصادر."
    )


# عرض سجل المحادثة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# عرض مصادر آخر عملية بحث
if st.session_state.last_sources:
    with st.expander("🔗 مصادر وروابط البحث"):
        for source in st.session_state.last_sources:
            st.markdown(
                f"- [{source['title']}]({source['url']})"
            )


user_text = st.text_area(
    "اكتب سؤالك",
    placeholder="اسأل عن أي موضوع...",
    height=120,
    key="current_question"
)

uploaded_image = st.file_uploader(
    "أرفق صورة (اختياري)",
    type=["png", "jpg", "jpeg", "webp"],
    key="current_image"
)

if uploaded_image:
    st.image(
        uploaded_image,
        caption="الصورة المرفقة",
        use_container_width=True
    )


if st.button(
    "إرسال",
    type="primary",
    use_container_width=True
):
    if not user_text.strip() and uploaded_image is None:
        st.warning("اكتب سؤالًا أو أرفق صورة أولًا.")

    else:
        question = user_text.strip()

        if uploaded_image is not None:
            saved_question = (
                (question + "\n\n") if question else ""
            ) + "[أرفق المستخدم صورة]"
        else:
            saved_question = question

        try:
            with st.spinner("جارٍ تجهيز الإجابة..."):
                if uploaded_image is not None:
                    answer = analyze_image(
                        question,
                        uploaded_image
                    )
                    sources = []

                elif enable_web_search:
                    answer, sources = research_with_web(question)

                else:
                    answer = call_chat(question)
                    sources = []

            # حفظ السؤال والجواب في ذاكرة الجلسة
            st.session_state.messages.append({
                "role": "user",
                "content": saved_question
            })

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            st.session_state.last_sources = sources

            st.rerun()

        except Exception as e:
            st.error(f"حدث خطأ: {e}")
            st.info(
                "إذا كان الخطأ متعلقًا بالنموذج، تحقق من ظهوره "
                "في قائمة نماذج Groq ومن حدود الاستخدام."
            )
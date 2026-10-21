import os
import io
import json
import base64

import streamlit as st
from PIL import Image
from groq import Groq
from openai import OpenAI
import streamlit.components.v1 as components


# =========================
# الإعدادات
# =========================

st.set_page_config(
    page_title="مساعد الذكاء الاصطناعي",
    page_icon="🤖",
    layout="wide"
)

MAX_TOKENS = 512

SYSTEM_PROMPT = """
أنت مساعد ذكاء اصطناعي عربي دقيق ومباشر.

القواعد:
- لا تخترع معلومات أو مصادر.
- إذا لم تكن متأكدًا، وضّح درجة عدم اليقين.
- فرّق بين الحقائق والفرضيات.
- أجب عن السؤال مباشرة دون حشو.
- عند وجود خطأ في السؤال، صححه باختصار.
- استخدم العربية الواضحة، إلا إذا طلب المستخدم لغة أخرى.
"""


# =========================
# التصميم
# =========================

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


# =========================
# مفاتيح API
# =========================

def get_secret(name):
    value = os.getenv(name)

    if value:
        return value

    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


GROQ_API_KEY = get_secret("GROQ_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")


if not GROQ_API_KEY:
    st.error(
        "لم يتم العثور على GROQ_API_KEY. "
        "أضفه في Secrets داخل إعدادات التطبيق."
    )
    st.stop()


groq_client = Groq(api_key=GROQ_API_KEY)


# =========================
# جلب النماذج
# =========================

@st.cache_data(ttl=300)
def get_models(api_key):
    try:
        client = Groq(api_key=api_key)
        result = client.models.list()

        return sorted([
            model.id for model in result.data
        ])

    except Exception as e:
        st.error(f"تعذر جلب النماذج: {e}")
        return []


models = get_models(GROQ_API_KEY)


def model_is_vision(model_name):
    name = model_name.lower()

    vision_keywords = [
        "vision",
        "llama-4",
        "llama-3.2-11b",
        "llama-3.2-90b"
    ]

    return any(word in name for word in vision_keywords)


vision_models = [
    model for model in models
    if model_is_vision(model)
]

text_models = [
    model for model in models
    if not model_is_vision(model)
]

if not text_models:
    text_models = models


# =========================
# تحويل الصورة
# =========================

def image_to_data_url(image_file, max_side=768):
    image = Image.open(image_file).convert("RGB")

    image.thumbnail((max_side, max_side))

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=60,
        optimize=True
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


# =========================
# المحادثة النصية
# =========================

def call_chat(client, model, user_text):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_text
            }
        ],
        temperature=0.2,
        max_tokens=MAX_TOKENS
    )

    return response.choices[0].message.content


# =========================
# تحليل الصور
# =========================

def analyze_image(client, model, user_text, image_file):
    image_url = image_to_data_url(image_file)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_text or "حلل الصورة واشرح محتواها بدقة."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        temperature=0.2,
        max_tokens=MAX_TOKENS
    )

    return response.choices[0].message.content


# =========================
# البحث عبر الويب
# =========================

def research_with_web(question, model):
    if not OPENAI_API_KEY:
        return (
            "البحث عبر الويب يحتاج إلى OPENAI_API_KEY "
            "ومزود يدعم أداة البحث. يمكنك تعطيل خيار البحث "
            "والمتابعة بالمحادثة العادية."
        )

    try:
        client = OpenAI(
            api_key=OPENAI_API_KEY
        )

        response = client.responses.create(
            model=model,
            input=(
                SYSTEM_PROMPT
                + "\n\nأجب عن السؤال التالي مع التحقق من المعلومات:\n"
                + question
            ),
            tools=[
                {"type": "web_search"}
            ]
        )

        return response.output_text

    except Exception as e:
        return (
            "تعذر تنفيذ البحث عبر الويب. "
            "تحقق من دعم النموذج لأداة البحث وإعدادات API.\n\n"
            f"تفاصيل الخطأ: {e}"
        )


# =========================
# واجهة التطبيق
# =========================

st.title("🤖 مساعد الذكاء الاصطناعي")
st.caption("محادثة، تحليل صور، وإمكانية البحث عبر الويب")

if not models:
    st.error(
        "لم أستطع جلب نماذج Groq. تحقق من مفتاح API."
    )
    st.stop()


with st.sidebar:
    st.header("⚙️ الإعدادات")

    selected_text_model = st.selectbox(
        "نموذج المحادثة",
        options=text_models
    )

    if vision_models:
        selected_vision_model = st.selectbox(
            "نموذج تحليل الصور",
            options=vision_models
        )
    else:
        selected_vision_model = None
        st.warning(
            "لم يُعثر على نموذج صور بالاعتماد على اسم النموذج. "
            "قد تحتاج إلى اختيار نموذج يدعم الصور يدويًا."
        )

    enable_web_search = st.toggle(
        "تفعيل البحث عبر الويب",
        value=False
    )

    st.caption(
        f"حد الإخراج الحالي: {MAX_TOKENS} توكن"
    )


user_text = st.text_area(
    "اكتب سؤالك",
    placeholder="اسأل عن أي موضوع...",
    height=140
)

uploaded_image = st.file_uploader(
    "أرفق صورة للتحليل (اختياري)",
    type=["png", "jpg", "jpeg", "webp"]
)

if uploaded_image:
    st.image(
        uploaded_image,
        caption="الصورة المرفقة",
        use_container_width=True
    )


if st.button("إرسال", type="primary", use_container_width=True):
    if not user_text.strip() and not uploaded_image:
        st.warning("اكتب سؤالًا أو أرفق صورة أولًا.")

    else:
        try:
            with st.spinner("جارٍ تجهيز الإجابة..."):

                if uploaded_image:
                    if not selected_vision_model:
                        st.error(
                            "لا يوجد نموذج صور محدد. "
                            "اختر نموذجًا يدعم الصور من الإعدادات."
                        )
                        st.stop()

                    answer = analyze_image(
                        groq_client,
                        selected_vision_model,
                        user_text,
                        uploaded_image
                    )

                elif enable_web_search:
                    answer = research_with_web(
                        user_text,
                        selected_text_model
                    )

                else:
                    answer = call_chat(
                        groq_client,
                        selected_text_model,
                        user_text
                    )

            st.subheader("الإجابة")
            st.markdown(answer)

        except Exception as e:
            st.error(f"حدث خطأ أثناء تنفيذ الطلب: {e}")
            st.info(
                "تحقق من النموذج المحدد، ومفتاح API، "
                "وحدود النموذج المدعوم."
            )
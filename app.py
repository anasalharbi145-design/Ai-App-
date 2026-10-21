
import os
import io
import json
import base64

import streamlit as st
from PIL import Image
from groq import Groq
from openai import OpenAI
import streamlit.components.v1 as components


# =====================================================
# 1. إعداد الصفحة
# =====================================================

st.set_page_config(
    page_title="Nova AI",
    page_icon="🧠",
    layout="wide",
)

# =====================================================
# 2. تنسيق الواجهة
# =====================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #0e1117;
        color: #f5f5f5;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .nova-title {
        font-size: 2.1rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .nova-subtitle {
        text-align: center;
        color: #aeb6c2;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextInput"] input {
        background-color: #ffffff !important;
        color: #111111 !important;
        -webkit-text-fill-color: #111111 !important;
        caret-color: #111111 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
    }

    div[data-testid="stTextArea"] textarea::placeholder,
    div[data-testid="stTextInput"] input::placeholder {
        color: #666666 !important;
        -webkit-text-fill-color: #666666 !important;
        opacity: 1 !important;
    }

    div.stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 2.7rem;
    }

    [data-testid="stSidebar"] {
        background: #11151d;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# 3. تعليمات المساعد
# =====================================================

SYSTEM_PROMPT = """
أنت Nova AI، مساعد علمي وتحليلي دقيق.

قواعد الإجابة:

1. لا تختلق حقائق أو دراسات أو مصادر أو روابط.
2. لا تدّعِ أنك بحثت في الإنترنت إذا لم تستخدم أداة بحث فعلية.
3. فرّق بين الحقيقة المدعومة والفرضية والاستنتاج والتخمين.
4. إذا لم تعرف الإجابة، قل إنك لا تعرف.
5. تحقق من الأرقام والوحدات والحسابات قبل عرضها.
6. لا تعتبر الارتباط الإحصائي دليلًا على السببية.
7. لا تعمم نتائج دراسة واحدة على جميع الحالات.
8. وضّح حدود الأدلة عندما تكون مهمة.
9. فضّل الدراسات الأصلية والمصادر الرسمية الموثوقة.
10. لا تخترع أسماء مؤلفين أو مجلات أو تواريخ نشر.
11. إذا تعذر التحقق من مصدر، وضّح ذلك.
12. لا تصف فرضية بأنها حقيقة محسومة دون أدلة كافية.
13. عند تقييم معلومة، حدّد الأجزاء الصحيحة والخاطئة كلًّا على حدة.
14. إذا كانت المعلومة حديثة، وضّح حدود معرفتك إن لم يتوفر بحث.
15. عند تحليل الصور، صف ما يظهر فعلًا ولا تستنتج تفاصيل
    لا تدعمها الصورة.
16. لا تدّعِ أنك قرأت نصًا غير واضح في الصورة.
17. عند وجود شك، اذكره بدل تقديم التخمين على أنه حقيقة.
18. لا تبالغ في الثقة أو تستخدم ضمانات غير مبررة.
19. كن مباشرًا وواضحًا، وتوسع عندما يحتاج السؤال ذلك.
20. ضع المصادر في نهاية الإجابة إذا كانت متوفرة فعلًا.
"""


# =====================================================
# 4. مفتاح API
# =====================================================

def get_api_key():
    key = os.getenv("GROQ_API_KEY", "").strip()

    if key:
        return key

    try:
        return str(
            st.secrets.get("GROQ_API_KEY", "")
        ).strip()
    except Exception:
        return ""


# =====================================================
# 5. الاتصال بـ Groq
# =====================================================

def get_groq_client(api_key):
    return Groq(api_key=api_key)


def fetch_model_ids(client):
    try:
        result = client.models.list()
        return sorted({
            model.id
            for model in result.data
            if getattr(model, "id", None)
        })
    except Exception as exc:
        st.warning(f"تعذر تحميل النماذج: {exc}")
        return []


def choose_default_model(model_ids, keywords):
    for keyword in keywords:
        for model_id in model_ids:
            if keyword.lower() in model_id.lower():
                return model_id

    return model_ids[0] if model_ids else ""


def model_is_vision(model_id):
    name = model_id.lower()

    vision_keywords = [
        "vision",
        "llava",
        "pixtral",
        "qwen-vl",
        "maverick",
        "scout",
    ]

    return any(word in name for word in vision_keywords)


# =====================================================
# 6. تجهيز الصورة وتقليل حجم الطلب
# =====================================================

def image_to_data_url(uploaded_file, max_side=768):
    """
    تصغير الصورة وضغطها قبل إرسالها إلى النموذج.
    يساعد ذلك على تقليل حجم الطلب، لكنه لا يضمن أن
    النموذج يقبل الصور أو أن الطلب ضمن حدود الخدمة.
    """

    image = Image.open(uploaded_file).convert("RGB")
    image.thumbnail((max_side, max_side))

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=60,
        optimize=True,
    )

    image_bytes = buffer.getvalue()

    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


# =====================================================
# 7. المحادثة النصية
# =====================================================

def call_chat(client, model, user_text):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_text,
            },
        ],
        temperature=0.2,
        max_tokens=2500,
    )

    return (
        response.choices[0].message.content
        or "لم يُرجع النموذج إجابة."
    )


# =====================================================
# 8. تحليل الصور
# =====================================================

def analyze_image(client, model, user_text, image_data_url):
    prompt = user_text.strip()

    if not prompt:
        prompt = (
            "حلل الصورة واشرح ما يظهر فيها بدقة. "
            "ميّز بين ما يمكن ملاحظته وما لا يمكن الجزم به."
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                        },
                    },
                ],
            },
        ],
        temperature=0.2,
        max_tokens=2000,
    )

    return (
        response.choices[0].message.content
        or "لم يُرجع النموذج إجابة."
    )


# =====================================================
# 9. البحث على الويب
# =====================================================

def research_with_web(api_key, model, user_text):
    """
    يعتمد نجاح البحث على دعم واجهة Responses API وأداة
    browser_search في الخدمة والنموذج المحددين.
    """

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=(
            "تحقق من السؤال التالي باستخدام البحث على الويب. "
            "لا تختلق مصادر، واذكر الروابط التي استخدمتها فعلًا.\n\n"
            + user_text
        ),
        tools=[
            {"type": "browser_search"}
        ],
    )

    answer = getattr(response, "output_text", "") or ""

    sources = []

    try:
        for item in response.output:
            for content in getattr(item, "content", []) or []:
                for annotation in (
                    getattr(content, "annotations", []) or []
                ):
                    url = getattr(annotation, "url", None)
                    title = getattr(annotation, "title", None)

                    if url and not any(
                        source["url"] == url for source in sources
                    ):
                        sources.append({
                            "title": title or url,
                            "url": url,
                        })
    except Exception:
        pass

    if not answer.strip():
        answer = "لم تُرجع أداة البحث نصًا."

    return answer, sources


# =====================================================
# 10. زر نسخ الإجابة
# =====================================================

def render_copy_button(text_to_copy):
    safe_text = json.dumps(
        text_to_copy,
        ensure_ascii=False,
    )

    html = f"""
    <button id="copy-answer"
        style="
            background:#263449;
            color:white;
            border:1px solid #52637a;
            padding:9px 16px;
            border-radius:10px;
            cursor:pointer;
            font-weight:700;
        ">
        📋 نسخ الإجابة
    </button>

    <script>
    const answerText = {safe_text};

    document.getElementById("copy-answer").onclick =
        async function() {{
            try {{
                await navigator.clipboard.writeText(answerText);
                this.innerText = "تم النسخ";
            }} catch (error) {{
                this.innerText = "تعذر النسخ من المتصفح";
            }}
        }};
    </script>
    """

    components.html(html, height=48)


# =====================================================
# 11. عرض المصادر
# =====================================================

def show_sources(sources):
    with st.expander("📚 المصادر والمراجع", expanded=False):
        if not sources:
            st.caption(
                "لم تُستخرج روابط مصادر من الاستجابة. "
                "هذا لا يعني أن الإجابة تحققت من مصادر."
            )
            return

        for number, source in enumerate(sources, start=1):
            title = source.get("title") or source["url"]
            url = source["url"]

            st.markdown(f"{number}. [{title}]({url})")


# =====================================================
# 12. حالة التطبيق
# =====================================================

if "nova_answer" not in st.session_state:
    st.session_state.nova_answer = ""

if "nova_sources" not in st.session_state:
    st.session_state.nova_sources = []

if "nova_error" not in st.session_state:
    st.session_state.nova_error = ""


# =====================================================
# 13. عنوان التطبيق
# =====================================================

st.markdown(
    '<div class="nova-title">🧠 Nova AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="nova-subtitle">'
    'مساعد للتحليل والتعلّم والتحقق من المعلومات'
    '</div>',
    unsafe_allow_html=True,
)


# =====================================================
# 14. إعداد الاتصال والنماذج
# =====================================================

api_key = get_api_key()

if not api_key:
    st.warning(
        "أضف GROQ_API_KEY في Streamlit Secrets "
        "أو أدخل المفتاح أدناه."
    )

    api_key = st.text_input(
        "مفتاح Groq API",
        type="password",
        placeholder="ألصق المفتاح هنا",
    ).strip()


client = None
model_ids = []

if api_key:
    try:
        client = get_groq_client(api_key)
        model_ids = fetch_model_ids(client)
    except Exception as exc:
        st.error(f"تعذر الاتصال بـ Groq: {exc}")


# =====================================================
# 15. الإعدادات الجانبية
# =====================================================

with st.sidebar:
    st.title("⚙️ الإعدادات")

    if model_ids:
        default_text = choose_default_model(
            model_ids,
            ["llama", "qwen", "gpt-oss"],
        )

        text_model = st.selectbox(
            "نموذج المحادثة",
            model_ids,
            index=(
                model_ids.index(default_text)
                if default_text in model_ids else 0
            ),
        )

        vision_models = [
            model for model in model_ids
            if model_is_vision(model)
        ]

        vision_options = vision_models or model_ids

        vision_model = st.selectbox(
            "نموذج الصور",
            vision_options,
        )

        search_model = st.selectbox(
            "نموذج البحث",
            model_ids,
            index=(
                model_ids.index(default_text)
                if default_text in model_ids else 0
            ),
        )

    else:
        text_model = ""
        vision_model = ""
        search_model = ""

        st.info("تعذر تحميل النماذج. تحقق من مفتاح API.")

    st.divider()

    use_web_search = st.toggle(
        "🔎 البحث على الويب",
        value=False,
        help=(
            "قد لا تعمل أداة البحث إذا لم تدعمها "
            "الواجهة المستخدمة."
        ),
    )

    st.caption(
        "اختيار نموذج لا يضمن دعمه للصور أو البحث."
    )


# =====================================================
# 16. حقل السؤال وإرفاق الصورة
# =====================================================

st.subheader("💬 اكتب سؤالك")

user_prompt = st.text_area(
    "السؤال",
    placeholder="اكتب سؤالك هنا...",
    height=130,
    label_visibility="collapsed",
    key="nova_prompt",
)

upload_col, send_col = st.columns([4, 1])

with upload_col:
    uploaded_image = st.file_uploader(
        "📎 إرفاق صورة",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
    )

with send_col:
    st.markdown(
        "<div style='height: 27px'></div>",
        unsafe_allow_html=True,
    )

    send_clicked = st.button(
        "إرسال ➤",
        type="primary",
        use_container_width=True,
    )


if uploaded_image is not None:
    st.image(
        uploaded_image,
        caption="الصورة المرفقة",
        use_container_width=True,
    )


# =====================================================
# 17. معالجة الطلب
# =====================================================

if send_clicked:
    st.session_state.nova_answer = ""
    st.session_state.nova_sources = []
    st.session_state.nova_error = ""

    if not api_key:
        st.session_state.nova_error = (
            "أدخل مفتاح Groq API أولًا."
        )

    elif not user_prompt.strip() and uploaded_image is None:
        st.session_state.nova_error = (
            "اكتب سؤالًا أو أرفق صورة."
        )

    elif client is None:
        st.session_state.nova_error = (
            "تعذر الاتصال بالخدمة."
        )

    else:
        try:
            with st.spinner("جاري معالجة طلبك..."):

                if uploaded_image is not None:
                    if not vision_model:
                        raise ValueError(
                            "لم يتم اختيار نموذج للصور."
                        )

                    image_data_url = image_to_data_url(
                        uploaded_image,
                        max_side=768,
                    )

                    answer = analyze_image(
                        client,
                        vision_model,
                        user_prompt,
                        image_data_url,
                    )

                    sources = []

                elif use_web_search:
                    if not search_model:
                        raise ValueError(
                            "لم يتم اختيار نموذج للبحث."
                        )

                    answer, sources = research_with_web(
                        api_key,
                        search_model,
                        user_prompt,
                    )

                else:
                    if not text_model:
                        raise ValueError(
                            "لم يتم اختيار نموذج للمحادثة."
                        )

                    answer = call_chat(
                        client,
                        text_model,
                        user_prompt,
                    )

                    sources = []

                st.session_state.nova_answer = answer
                st.session_state.nova_sources = sources

        except Exception as exc:
            st.session_state.nova_error = str(exc)


# =====================================================
# 18. عرض النتيجة
# =====================================================

if st.session_state.nova_error:
    st.error(
        "تعذر تنفيذ الطلب:\n\n"
        + st.session_state.nova_error
    )


if st.session_state.nova_answer:
    st.divider()
    st.subheader("🧠 الإجابة")

    st.markdown(st.session_state.nova_answer)

    render_copy_button(
        st.session_state.nova_answer
    )

    st.download_button(
        "⬇️ تنزيل الإجابة",
        data=st.session_state.nova_answer,
        file_name="nova_answer.txt",
        mime="text/plain",
    )

    show_sources(
        st.session_state.nova_sources
    )

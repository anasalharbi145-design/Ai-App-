
import os
import re
import io
import json
import base64
from typing import Optional

import streamlit as st
from PIL import Image
from groq import Groq
from openai import OpenAI
import streamlit.components.v1 as components


# =========================================================
# 1. إعداد الصفحة
# =========================================================

st.set_page_config(
    page_title="Nova AI | المساعد العلمي",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. تنسيق الواجهة
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #0e1117;
        color: #f5f5f5;
    }

    [data-testid="stHeader"] {
        background: rgba(14, 17, 23, 0.95);
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

    /* اجعل حقول الكتابة بخلفية بيضاء ونص أسود */
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

    div[data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    div.stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 2.7rem;
    }

    .answer-box {
        padding: 1rem;
        border: 1px solid #303744;
        border-radius: 14px;
        background: #151a23;
    }

    [data-testid="stSidebar"] {
        background: #11151d;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. تعليمات المساعد
# =========================================================

SYSTEM_PROMPT = r"""
أنت Nova AI، مساعد علمي وتحليلي دقيق، تتحدث بالعربية الواضحة
وتستطيع الإجابة بلغات أخرى عند الحاجة.

هدفك تقديم إجابات صحيحة، والتحقق من الادعاءات، وتجنب اختلاق
المعلومات أو المراجع أو الروابط.

قواعد أساسية:

1. لا تخترع مصدرًا أو دراسة أو مؤلفًا أو تاريخ نشر أو رابطًا.
2. لا تدّعِ أنك بحثت في الإنترنت إذا لم تستخدم أداة بحث فعلية.
3. ميّز بين الحقيقة المدعومة، والاستنتاج، والفرضية، والمعلومة
   غير المؤكدة.
4. إذا كانت المعلومة غير معروفة لك، فقل ذلك بوضوح.
5. إذا كان السؤال يتضمن ادعاءً علميًا، فافحص دقته قبل تأييده.
6. لا تعتبر انتشار المعلومة دليلًا على صحتها.
7. لا تخلط بين الارتباط الإحصائي والسببية.
8. لا تعمم نتائج دراسة واحدة على جميع الحالات.
9. وضّح حجم العينة والمنهج والقيود إذا كانت متاحة ومهمة.
10. لا تقل إن دراسة أثبتت شيئًا إذا كانت لا تدعم إلا احتمالًا
    أو ارتباطًا أو نتيجة محدودة.
11. تحقق من الوحدات والأرقام والنسب والحسابات قبل عرضها.
12. عند وجود خلاف علمي معتبر، اذكر جوانب الخلاف باختصار.
13. فرّق بين نتائج التجارب، والملاحظات، والنماذج النظرية.
14. لا تصف فرضية بأنها حقيقة محسومة دون دليل كافٍ.
15. لا تخلط بين غياب الدليل والدليل على الغياب.
16. استخدم مصادر أولية قدر الإمكان، مثل الأوراق العلمية
    والمواقع الرسمية وقواعد البيانات العلمية.
17. عند تقييم دراسة، افحص عنوانها ومؤلفيها وسنتها ومجلة النشر
    والرابط المباشر متى أمكن.
18. إذا تعذر التحقق من مرجع، فلا تقدمه بوصفه مرجعًا متحققًا.
19. لا تضع روابط عامة بدل رابط الدراسة المقصودة إذا توفر رابطها.
20. لا تستخدم اقتباسًا حرفيًا طويلًا؛ لخّص بلغتك.
21. إذا طلب المستخدم تقييم نص، فحدد الأخطاء الفعلية وأسبابها،
    ولا تغيّر أجزاء صحيحة بلا سبب.
22. إذا كانت بعض المعلومات صحيحة وبعضها خاطئة، فافصل بينها.
23. عند تصحيح خطأ سابق، اذكر الخطأ والتصحيح بوضوح.
24. لا تبالغ في الثقة، ولا تستخدم كلمات مثل «مستحيل» أو
    «مؤكد 100%» إلا إذا كان لها مبرر مناسب.
25. إذا لم تتوفر إمكانية بحث حقيقية، وضّح أن التحقق محدود.
26. لا تدّعِ أنك فتحت صفحة أو قرأت دراسة ما لم يحدث ذلك فعليًا.
27. في الإجابات العلمية، رتّب المحتوى حسب الحاجة:
    الحكم المختصر، التفسير، مستوى الثقة، ثم المصادر.
28. اجعل المصادر في نهاية الإجابة، ولا تكررها في كل فقرة
    بلا داعٍ.
29. لا تعرض مراجع غير مستخدمة لدعم الادعاءات.
30. راجع الإجابة قبل إرسالها: هل توجد أرقام غير مبررة؟
    هل توجد قفزة استنتاجية؟ هل المصدر يدعم الادعاء فعلًا؟
31. إذا طلب المستخدم معلومات نادرة، فلا تجعل الندرة أهم من
    الصحة. قدّم معلومات قابلة للتحقق وتجنب الغرائب الملفقة.
32. عند تلخيص مصدر، التزم بما يقوله المصدر ولا تضف إليه
    استنتاجات على أنها نتائج أصلية.
33. إذا كانت المعلومة حديثة أو قابلة للتغير، فاذكر تاريخ
    التحقق عند توفره.
34. لا تزعم أن لديك أدوات أو قدرات لم تُتح لك فعليًا.
35. كن مباشرًا ومختصرًا افتراضيًا، وتوسع عند الحاجة.
36. لا تضع قسم مصادر وهميًا إذا لم تتوفر مصادر حقيقية.

تنسيق المصادر:
ضع المصادر في نهاية الإجابة، مع اسم الجهة أو الدراسة والرابط
المباشر متى توفر. افصل بوضوح بين المصادر التي تم التحقق منها
والمصادر المقترحة التي لم يتم التحقق منها.
"""


# =========================================================
# 4. قراءة مفتاح API
# =========================================================

def get_api_key() -> str:
    """قراءة المفتاح من متغيرات البيئة أو إعدادات Streamlit."""
    key = os.getenv("GROQ_API_KEY", "").strip()

    if key:
        return key

    try:
        key = str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except Exception:
        key = ""

    return key


# =========================================================
# 5. إنشاء عميل Groq
# =========================================================

def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


# =========================================================
# 6. جلب قائمة النماذج المتاحة
# =========================================================

def fetch_model_ids(client: Groq) -> list[str]:
    try:
        response = client.models.list()
        model_ids = []

        for model in response.data:
            model_id = getattr(model, "id", None)
            if model_id:
                model_ids.append(model_id)

        return sorted(set(model_ids))

    except Exception as exc:
        st.warning(f"تعذر تحميل قائمة النماذج تلقائيًا: {exc}")
        return []


def choose_default_model(
    model_ids: list[str],
    keywords: list[str],
    fallback: str = "",
) -> str:
    """اختيار نموذج مناسب من النماذج التي أعادها الخادم."""
    if not model_ids:
        return fallback

    for keyword in keywords:
        for model_id in model_ids:
            if keyword.lower() in model_id.lower():
                return model_id

    return model_ids[0]


def model_is_vision(model_id: str) -> bool:
    """تخمين مبدئي من اسم النموذج؛ لا يضمن دعم الصور."""
    model_id = model_id.lower()

    keywords = [
        "vision",
        "llava",
        "pixtral",
        "qwen-vl",
        "maverick",
        "scout",
    ]

    return any(word in model_id for word in keywords)


# =========================================================
# 7. تجهيز الصور
# =========================================================

def image_to_data_url(
    uploaded_file,
    max_side: int = 1600,
) -> str:
    """
    تحويل الصورة إلى JPEG مضغوط ثم إلى Data URL.
    يهدف ذلك إلى تقليل حجم الطلب.
    """
    image = Image.open(uploaded_file).convert("RGB")

    image.thumbnail((max_side, max_side))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


# =========================================================
# 8. استخراج المصادر من استجابة Responses API
# =========================================================

def extract_sources(response) -> list[dict]:
    """
    استخراج المصادر إذا أعادها النموذج في annotations.
    إذا لم تتوفر annotations فلن يخترع البرنامج مصادر.
    """
    sources = []
    seen_urls = set()

    try:
        output_items = getattr(response, "output", []) or []

        for item in output_items:
            content_items = getattr(item, "content", []) or []

            for content in content_items:
                annotations = getattr(content, "annotations", []) or []

                for annotation in annotations:
                    url = getattr(annotation, "url", None)
                    title = getattr(annotation, "title", None)

                    if not url:
                        continue

                    if url in seen_urls:
                        continue

                    seen_urls.add(url)

                    sources.append(
                        {
                            "title": title or url,
                            "url": url,
                        }
                    )

    except Exception:
        pass

    return sources


# =========================================================
# 9. استدعاء نموذج المحادثة
# =========================================================

def call_chat(
    client: Groq,
    model: str,
    user_text: str,
    history: Optional[list] = None,
) -> str:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    if history:
        messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": user_text,
        }
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    return answer or "لم يُرجع النموذج نصًا."


# =========================================================
# 10. تحليل الصور
# =========================================================

def analyze_image(
    client: Groq,
    model: str,
    user_text: str,
    image_data_url: str,
) -> str:
    """
    يتطلب نموذجًا يدعم الصور عبر واجهة chat.completions.
    قد تختلف النماذج المتاحة ودعم الصور حسب حساب Groq.
    """
    prompt = user_text.strip()

    if not prompt:
        prompt = (
            "حلل الصورة بدقة. اذكر ما يمكن ملاحظته مباشرة، "
            "وما لا يمكن الجزم به من الصورة وحدها."
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
    )

    return response.choices[0].message.content or "لم يُرجع النموذج إجابة."


# =========================================================
# 11. البحث على الويب
# =========================================================

def research_with_web(
    api_key: str,
    model: str,
    user_text: str,
) -> tuple[str, list[dict]]:
    """
    يحاول استخدام أداة البحث عبر Responses API.
    دعم browser_search يعتمد على الواجهة والنموذج المستخدمين.
    إذا لم يكن مدعومًا، يعرض الخطأ بدل ادعاء نجاح البحث.
    """
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=(
            "ابحث عن معلومات موثوقة للتحقق من السؤال التالي. "
            "لا تخترع مصادر، واستشهد بالمصادر التي استخدمتها فعلًا.\n\n"
            + user_text
        ),
        tools=[
            {
                "type": "browser_search",
            }
        ],
    )

    answer = getattr(response, "output_text", "") or ""
    sources = extract_sources(response)

    return answer, sources


# =========================================================
# 12. زر نسخ الإجابة
# =========================================================

def render_copy_button(text_to_copy: str):
    """زر نسخ يعمل إذا سمحت بيئة المتصفح بالوصول إلى الحافظة."""
    safe_text = json.dumps(text_to_copy, ensure_ascii=False)

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

    document.getElementById("copy-answer").onclick = async () => {{
        try {{
            await navigator.clipboard.writeText(answerText);
            document.getElementById("copy-answer").innerText = "تم النسخ";
        }} catch (error) {{
            document.getElementById("copy-answer").innerText =
                "تعذر النسخ من هذه البيئة";
        }}
    }};
    </script>
    """

    components.html(html, height=48)


# =========================================================
# 13. عرض المصادر
# =========================================================

def show_sources(sources: list[dict]):
    with st.expander("📚 المصادر والمراجع", expanded=False):
        if not sources:
            st.caption(
                "لم تُستخرج روابط مصادر من الاستجابة. "
                "لا يعني ذلك أن الإجابة تحققت من المصادر."
            )
            return

        for index, source in enumerate(sources, start=1):
            title = source.get("title") or source.get("url")
            url = source.get("url")

            if url:
                st.markdown(f"{index}. [{title}]({url})")


# =========================================================
# 14. تهيئة حالة التطبيق
# =========================================================

if "nova_answer" not in st.session_state:
    st.session_state["nova_answer"] = ""

if "nova_sources" not in st.session_state:
    st.session_state["nova_sources"] = []

if "nova_error" not in st.session_state:
    st.session_state["nova_error"] = ""


# =========================================================
# 15. واجهة التطبيق
# =========================================================

st.markdown(
    '<div class="nova-title">🧠 Nova AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="nova-subtitle">'
    'مساعدك للتحليل والتعلّم والتحقق من المعلومات'
    '</div>',
    unsafe_allow_html=True,
)


# =========================================================
# 16. المفتاح والنماذج
# =========================================================

api_key = get_api_key()

if not api_key:
    st.warning(
        "أضف مفتاح GROQ_API_KEY في إعدادات Secrets أو "
        "في متغيرات البيئة لتشغيل المساعد."
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
        st.error(f"تعذر الاتصال بخدمة Groq: {exc}")


# =========================================================
# 17. الشريط الجانبي
# =========================================================

with st.sidebar:
    st.title("⚙️ الإعدادات")

    st.caption("اختر النموذج المناسب من النماذج المتاحة لحسابك.")

    if model_ids:
        text_default = choose_default_model(
            model_ids,
            ["llama", "qwen", "gpt-oss"],
            model_ids[0],
        )

        text_index = (
            model_ids.index(text_default)
            if text_default in model_ids
            else 0
        )

        text_model = st.selectbox(
            "نموذج المحادثة",
            options=model_ids,
            index=text_index,
        )

        vision_models = [
            model for model in model_ids
            if model_is_vision(model)
        ]

        if vision_models:
            vision_model = st.selectbox(
                "نموذج الصور",
                options=vision_models,
            )
        else:
            vision_model = st.selectbox(
                "نموذج الصور",
                options=model_ids,
                help=(
                    "اختر نموذجًا يدعم الصور. "
                    "ليس كل نموذج نصي قادرًا على تحليل الصور."
                ),
            )

        search_model = st.selectbox(
            "نموذج البحث",
            options=model_ids,
            index=text_index,
            help=(
                "قد يتطلب البحث واجهة تدعم أداة البحث. "
                "ظهور النموذج هنا لا يعني أن البحث مدعوم."
            ),
        )

    else:
        st.info(
            "ستظهر النماذج هنا بعد نجاح الاتصال "
            "وتوفر قائمة النماذج."
        )

        text_model = ""
        vision_model = ""
        search_model = ""

    st.divider()

    use_web_search = st.toggle(
        "🔎 البحث على الويب",
        value=False,
        help=(
            "إذا كانت أداة البحث غير مدعومة في الواجهة أو النموذج، "
            "سيظهر الخطأ بدل ادعاء إتمام البحث."
        ),
    )

    st.caption(
        "ملاحظة: اختيار نموذج لا يضمن دعمه للصور أو البحث؛ "
        "ذلك يعتمد على قدرات النموذج والواجهة."
    )


# =========================================================
# 18. إدخال السؤال وإرفاق الصورة
# =========================================================

st.subheader("💬 اكتب سؤالك")

user_prompt = st.text_area(
    "السؤال",
    placeholder=(
        "اكتب سؤالك هنا... مثال: تحقق من صحة هذه المعلومة "
        "العلمية وبيّن الأدلة والقيود."
    ),
    height=130,
    label_visibility="collapsed",
    key="nova_prompt",
)

st.markdown("")

# مساحة رفع الصورة وزر الإرسال في الصف نفسه
upload_col, send_col = st.columns([4, 1])

with upload_col:
    uploaded_image = st.file_uploader(
        "📎 إرفاق صورة",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
        help="يمكنك إرفاق صورة لتحليلها إذا كان النموذج المختار يدعم الصور.",
    )

with send_col:
    st.markdown("<div style='height: 27px'></div>", unsafe_allow_html=True)

    send_clicked = st.button(
        "إرسال ➤",
        type="primary",
        use_container_width=True,
    )


# =========================================================
# 19. معاينة الصورة
# =========================================================

if uploaded_image is not None:
    st.image(
        uploaded_image,
        caption="الصورة المرفقة",
        use_container_width=True,
    )


# =========================================================
# 20. تنفيذ الطلب
# =========================================================

if send_clicked:
    st.session_state["nova_answer"] = ""
    st.session_state["nova_sources"] = []
    st.session_state["nova_error"] = ""

    if not api_key:
        st.session_state["nova_error"] = (
            "أدخل مفتاح Groq API أولًا."
        )

    elif not user_prompt.strip() and uploaded_image is None:
        st.session_state["nova_error"] = (
            "اكتب سؤالًا أو أرفق صورة أولًا."
        )

    elif client is None:
        st.session_state["nova_error"] = (
            "تعذر إنشاء اتصال بالخدمة."
        )

    else:
        try:
            with st.spinner("جاري معالجة طلبك..."):

                if uploaded_image is not None:
                    # تحليل الصورة
                    image_data_url = image_to_data_url(uploaded_image)

                    if not vision_model:
                        raise ValueError(
                            "لم يتم اختيار نموذج لتحليل الصور."
                        )

                    answer = analyze_image(
                        client=client,
                        model=vision_model,
                        user_text=user_prompt,
                        image_data_url=image_data_url,
                    )

                    sources = []

                elif use_web_search:
                    # البحث باستخدام Responses API
                    if not search_model:
                        raise ValueError(
                            "لم يتم اختيار نموذج للبحث."
                        )

                    answer, sources = research_with_web(
                        api_key=api_key,
                        model=search_model,
                        user_text=user_prompt,
                    )

                else:
                    # محادثة عادية دون ادعاء البحث
                    if not text_model:
                        raise ValueError(
                            "لم يتم اختيار نموذج للمحادثة."
                        )

                    answer = call_chat(
                        client=client,
                        model=text_model,
                        user_text=user_prompt,
                    )

                    sources = []

                st.session_state["nova_answer"] = answer
                st.session_state["nova_sources"] = sources

        except Exception as exc:
            st.session_state["nova_error"] = str(exc)


# =========================================================
# 21. عرض النتيجة والمصادر
# =========================================================

if st.session_state["nova_error"]:
    st.error(
        "تعذر تنفيذ الطلب:\n\n"
        + st.session_state["nova_error"]
    )

if st.session_state["nova_answer"]:
    st.divider()

    st.subheader("🧠 الإجابة")

    st.markdown(
        '<div class="answer-box">',
        unsafe_allow_html=True,
    )

    st.markdown(st.session_state["nova_answer"])

    st.markdown("</div>", unsafe_allow_html=True)

    render_copy_button(st.session_state["nova_answer"])

    st.download_button(
        label="⬇️ تنزيل الإجابة كملف نصي",
        data=st.session_state["nova_answer"],
        file_name="nova_answer.txt",
        mime="text/plain",
    )

    # المصادر في النهاية داخل قسم قابل للفتح
    show_sources(st.session_state["nova_sources"])

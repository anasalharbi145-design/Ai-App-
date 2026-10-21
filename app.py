
import base64
import io
import re
import uuid
from datetime import datetime

import streamlit as st
from groq import Groq
from PIL import Image

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None


# ==================================================
# إعدادات التطبيق
# ==================================================

st.set_page_config(
    page_title="NOVA AI",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded",
)

SYSTEM_PROMPT = """
أنت مساعد ذكي باللغة العربية، دقيق وصريح.

في الأسئلة العلمية:
- افحص الادعاءات واحداً واحداً.
- ميّز بين الصحيح والصحيح جزئياً والخاطئ وغير الموثق.
- لا تختلق مراجع أو أسماء دراسات أو روابط.
- لا تقل إنك فتحت مصدراً إذا لم تفتحه فعلياً.
- فرّق بين الحقائق التجريبية والفرضيات والاستنتاجات.
- استخدم نتائج البحث بحذر، ولا تعتبر المقتطف وحده إثباتاً.
- إذا لم تتوفر أدلة كافية، قل ذلك بوضوح.
- صحح الأخطاء واشرح سبب التصحيح.
- أعط الأولوية للمصادر الأصلية والجهات العلمية الرسمية.
- لا تخترع درجات ثقة رقمية.
- إذا كان السؤال عادياً، أجب مباشرة دون تحويله إلى تقرير علمي.
- لا تطل الإجابة بلا حاجة.
"""

EXCLUDED_TERMS = (
    "whisper", "guard", "moderation", "embedding",
    "speech", "tts", "transcription", "safeguard",
)

VISION_HINTS = (
    "vision", "llama-4", "llama-3.2-90b",
    "llama-3.2-11b", "qwen", "maverick", "scout",
)


# ==================================================
# تصميم الواجهة
# ==================================================

st.markdown("""
<style>
:root {
    --bg: #212121;
    --panel: #171717;
    --panel2: #2f2f2f;
    --border: #3b3b3b;
    --text: #ececec;
    --muted: #a0a0a0;
    --accent: #d9d9d9;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stMain"] {
    background: var(--bg) !important;
}

[data-testid="stSidebar"] {
    background: var(--panel) !important;
    border-right: 1px solid #292929;
}

[data-testid="stSidebar"] > div:first-child {
    background: var(--panel) !important;
}

.block-container {
    max-width: 920px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 7rem !important;
}

h1, h2, h3, p, label, li {
    color: var(--text);
}

.nova-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 4px 22px 4px;
}

.nova-logo {
    width: 38px;
    height: 38px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #171717;
    background: #ececec;
    font-size: 24px;
    font-weight: 700;
}

.nova-name {
    font-size: 21px;
    font-weight: 650;
    letter-spacing: -0.6px;
    color: #f5f5f5;
}

.nova-subtitle {
    color: #929292;
    font-size: 12px;
    margin-top: 1px;
}

.hero {
    text-align: center;
    padding: 13vh 10px 20px 10px;
}

.hero-symbol {
    font-size: 36px;
    color: #e4e4e4;
    margin-bottom: 16px;
}

.hero-title {
    font-size: clamp(27px, 4vw, 38px);
    font-weight: 600;
    letter-spacing: -1px;
    color: #f4f4f4;
}

.hero-subtitle {
    color: #a2a2a2;
    font-size: 15px;
    margin-top: 12px;
}

div[data-testid="stChatMessage"] {
    background: transparent;
    border: none;
    padding: 14px 0;
}

[data-testid="stChatMessageContent"] {
    color: #ededed;
    line-height: 1.85;
}

[data-testid="stChatInput"] {
    background: #303030 !important;
    border: 1px solid #454545 !important;
    border-radius: 25px !important;
    box-shadow: 0 4px 25px rgba(0,0,0,.15);
}

[data-testid="stChatInput"] textarea {
    color: #f2f2f2 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #a0a0a0 !important;
}

[data-testid="stFileUploader"] {
    background: #292929;
    border: 1px dashed #505050;
    border-radius: 14px;
    padding: 8px;
}

.stButton button {
    border-radius: 12px;
    min-height: 40px;
    transition: all .15s ease;
}

.stButton button:hover {
    border-color: #aaaaaa;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: #353535 !important;
}

.small-muted {
    color: #8e8e8e;
    font-size: 12px;
}

[data-testid="stSidebar"] .stButton button {
    text-align: left;
    border: 1px solid transparent;
    background: transparent;
}

[data-testid="stSidebar"] .stButton button:hover {
    background: #292929;
    border-color: #353535;
}

@media (max-width: 700px) {
    .block-container {
        padding: 1rem 1rem 6rem 1rem !important;
    }

    .hero {
        padding-top: 8vh;
    }

    .hero-title {
        font-size: 27px;
    }
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# إدارة الجلسة والمحادثات
# ==================================================

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "active_chat" not in st.session_state:
    st.session_state.active_chat = None

if "pending_image" not in st.session_state:
    st.session_state.pending_image = None

if "last_search_results" not in st.session_state:
    st.session_state.last_search_results = {}


def new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "محادثة جديدة",
        "messages": [],
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    st.session_state.active_chat = chat_id


def get_active_chat():
    chat_id = st.session_state.active_chat

    if chat_id and chat_id in st.session_state.chats:
        return st.session_state.chats[chat_id]

    return None


# ==================================================
# الاتصال بالنموذج
# ==================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_model_ids(api_key):
    client = Groq(api_key=api_key)
    response = client.models.list()

    ids = []

    for model in response.data:
        model_id = getattr(model, "id", "")
        if not model_id:
            continue

        name = model_id.lower()

        if any(term in name for term in EXCLUDED_TERMS):
            continue

        ids.append(model_id)

    return sorted(set(ids), key=model_score, reverse=True)


def model_score(model_id):
    name = model_id.lower()

    ranking = [
        ("gpt-oss-120b", 100),
        ("gpt-oss-20b", 80),
        ("llama-4", 75),
        ("llama-3.3", 70),
        ("qwen", 65),
        ("llama-3.1", 60),
        ("llama-3.2", 55),
    ]

    score = 0

    for family, value in ranking:
        if family in name:
            score += value
            break

    sizes = re.findall(r"(\d+)\s*b", name)
    if sizes:
        score += min(int(sizes[0]), 500) / 10

    return score


def supports_vision(model_id):
    name = model_id.lower()
    return any(term in name for term in VISION_HINTS)


# ==================================================
# البحث على الويب
# ==================================================

def web_search(query, max_results=5):
    if DDGS is None or not query.strip():
        return []

    try:
        with DDGS() as search:
            results = list(search.text(query, max_results=max_results))

        return [
            {
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "snippet": result.get("body", ""),
            }
            for result in results
        ]

    except Exception:
        return []


def search_context(results):
    if not results:
        return "لم تتوفر نتائج بحث. لا تدّع أن البحث نجح."

    return "\n\n".join(
        f"العنوان: {r['title']}\n"
        f"الرابط: {r['url']}\n"
        f"المقتطف: {r['snippet']}"
        for r in results
    )


# ==================================================
# تجهيز الصور
# ==================================================

def make_image_data_url(upload):
    image = Image.open(upload).convert("RGB")
    image.thumbnail((1600, 1600))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


# ==================================================
# الشريط الجانبي
# ==================================================

with st.sidebar:
    st.markdown("""
    <div class="nova-brand">
        <div class="nova-logo">✳</div>
        <div>
            <div class="nova-name">NOVA AI</div>
            <div class="nova-subtitle">مساحتك الذكية</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("＋  محادثة جديدة", use_container_width=True):
        new_chat()
        st.rerun()

    st.divider()

    st.markdown("**محادثاتك**")

    chat_items = list(st.session_state.chats.items())
    chat_items.reverse()

    if not chat_items:
        st.markdown(
            '<div class="small-muted">محادثاتك الجديدة ستظهر هنا.</div>',
            unsafe_allow_html=True,
        )

    for chat_id, chat in chat_items:
        title = chat["title"][:32]
        if len(chat["title"]) > 32:
            title += "…"

        if st.button(
            f"▤  {title}",
            key=f"open_{chat_id}",
            use_container_width=True,
        ):
            st.session_state.active_chat = chat_id
            st.rerun()

    st.divider()

    with st.expander("الإعدادات", expanded=False):
        api_key = st.text_input(
            "مفتاح Groq API",
            value=st.secrets.get("GROQ_API_KEY", ""),
            type="password",
        )

        enable_search = st.toggle(
            "البحث على الويب",
            value=True,
        )

        st.caption(
            "لا تشارك مفتاح API مع أحد، ولا تضعه داخل الكود المنشور."
        )

    selected_model = None

    if api_key:
        try:
            model_ids = fetch_model_ids(api_key)

            if model_ids:
                selected_model = st.selectbox(
                    "نموذج الذكاء الاصطناعي",
                    model_ids,
                    index=0,
                )
            else:
                st.warning("لم تُرجع الخدمة قائمة نماذج.")
        except Exception as exc:
            st.error("تعذر تحميل النماذج.")
            with st.expander("تفاصيل الخطأ"):
                st.code(str(exc))

    st.markdown(
        '<div class="small-muted">قد تختلف النماذج المتاحة حسب حسابك.</div>',
        unsafe_allow_html=True,
    )


# ==================================================
# الصفحة الرئيسية
# ==================================================

chat = get_active_chat()

if chat is None:
    st.markdown("""
    <div class="hero">
        <div class="hero-symbol">✳</div>
        <div class="hero-title">وش ودّك تعرف اليوم؟</div>
        <div class="hero-subtitle">
            اسأل، حلّل، ابحث، أو تحقّق من معلومة.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "🔬  تدقيق علمي",
            use_container_width=True,
        ):
            new_chat()
            st.session_state.chats[
                st.session_state.active_chat
            ]["messages"].append({
                "role": "assistant",
                "content": "أرسل المعلومة العلمية التي تريد التحقق منها.",
            })
            st.rerun()

    with col2:
        if st.button(
            "🧠  شرح موضوع",
            use_container_width=True,
        ):
            new_chat()
            st.session_state.chats[
                st.session_state.active_chat
            ]["messages"].append({
                "role": "assistant",
                "content": "ما الموضوع الذي تريد أن أشرحه لك؟",
            })
            st.rerun()

    with col3:
        if st.button(
            "🖼️  تحليل صورة",
            use_container_width=True,
        ):
            new_chat()
            st.session_state.chats[
                st.session_state.active_chat
            ]["messages"].append({
                "role": "assistant",
                "content": "ارفع صورة باستخدام زر الإرفاق أسفل المحادثة.",
            })
            st.rerun()

    chat = get_active_chat()


if chat is not None:
    for message in chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("image_preview"):
                st.image(
                    message["image_preview"],
                    use_container_width=True,
                )

            if message["role"] == "assistant":
                if message.get("sources"):
                    with st.expander("المصادر التي عُثر عليها"):
                        for source in message["sources"]:
                            st.markdown(
                                f"**{source['title']}**\n\n"
                                f"{source['snippet']}\n\n"
                                f"[فتح الرابط]({source['url']})"
                            )
                            st.divider()

                st.button(
                    "نسخ الرد",
                    key=f"copy_{message.get('id', uuid.uuid4())}",
                    on_click=lambda value=message["content"]: (
                        st.session_state.update(
                            copy_buffer=value
                        )
                    ),
                )

    if "copy_buffer" in st.session_state:
        st.text_area(
            "انسخ الرد من هنا",
            value=st.session_state.copy_buffer,
            height=120,
        )


# ==================================================
# إدخال الرسائل والمرفقات
# ==================================================

if chat is not None:
    with st.expander("＋ إرفاق صورة", expanded=False):
        uploaded_file = st.file_uploader(
            "اختر صورة للتحليل",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"upload_{st.session_state.active_chat}",
        )

        if uploaded_file:
            st.image(
                uploaded_file,
                caption="الصورة المرفقة",
                use_container_width=True,
            )

    prompt = st.chat_input(
        "اكتب رسالتك...",
        key=f"prompt_{st.session_state.active_chat}",
    )

    if prompt:
        if not api_key or not selected_model:
            st.error("أدخل مفتاح Groq صالحاً واختر نموذجاً.")
            st.stop()

        if uploaded_file and not supports_vision(selected_model):
            st.error(
                "قد لا يدعم النموذج المختار الصور. "
                "اختر نموذجاً يدعم الرؤية."
            )
            st.stop()

        current_chat = get_active_chat()

        if current_chat is None:
            new_chat()
            current_chat = get_active_chat()

        if current_chat["title"] == "محادثة جديدة":
            current_chat["title"] = prompt[:32]

        user_message = {
            "role": "user",
            "content": prompt,
            "id": str(uuid.uuid4()),
        }

        if uploaded_file:
            user_message["image_preview"] = uploaded_file.getvalue()

        current_chat["messages"].append(user_message)

        with st.chat_message("user"):
            st.markdown(prompt)

            if uploaded_file:
                st.image(
                    uploaded_file,
                    use_container_width=True,
                )

        with st.chat_message("assistant"):
            with st.spinner("أفكر في إجابتك..."):
                try:
                    client = Groq(api_key=api_key)

                    history = []

                    for old_message in current_chat["messages"][-13:]:
                        if old_message["role"] not in ("user", "assistant"):
                            continue

                        content = old_message["content"]

                        if old_message["role"] == "user":
                            history.append({
                                "role": "user",
                                "content": content,
                            })
                        else:
                            history.append({
                                "role": "assistant",
                                "content": content,
                            })

                    sources = []

                    if enable_search:
                        with st.spinner("أبحث عن مصادر..."):
                            query = prompt[:400]
                            sources = web_search(query)

                    context = search_context(sources)

                    enhanced_prompt = (
                        f"{prompt}\n\n"
                        "نتائج البحث المتاحة:\n"
                        f"{context}\n\n"
                        "استخدم نتائج البحث بحذر. لا تقل إنك فتحت "
                        "صفحات لم تفتحها. إذا لم تكف الأدلة، "
                        "اذكر حدود التحقق بوضوح."
                    )

                    history[-1]["content"] = enhanced_prompt

                    if uploaded_file:
                        data_url = make_image_data_url(uploaded_file)

                        history[-1]["content"] = [
                            {
                                "type": "text",
                                "text": enhanced_prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                },
                            },
                        ]

                    response = client.chat.completions.create(
                        model=selected_model,
                        messages=[
                            {
                                "role": "system",
                                "content": SYSTEM_PROMPT,
                            },
                            *history,
                        ],
                        temperature=0.2,
                        max_tokens=5000,
                    )

                    answer = (
                        response.choices[0].message.content
                        or "لم يُرجع النموذج نصاً."
                    )

                    st.markdown(answer)

                    if sources:
                        with st.expander("المصادر التي عُثر عليها"):
                            for source in sources:
                                st.markdown(
                                    f"**{source['title']}**\n\n"
                                    f"{source['snippet']}\n\n"
                                    f"[فتح الرابط]({source['url']})"
                                )
                                st.divider()

                    current_chat["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "id": str(uuid.uuid4()),
                    })

                    st.rerun()

                except Exception as exc:
                    st.error(
                        "تعذر الحصول على الرد. "
                        "تحقق من النموذج ومفتاح API وحدود الاستخدام."
                    )
                    st.code(str(exc))

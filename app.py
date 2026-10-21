
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

MAX_TOKENS = 2048
HISTORY_LIMIT = 12

SYSTEM_PROMPT = """
أنت مساعد ذكاء اصطناعي عام متعدد المجالات.
هدفك تقديم إجابات دقيقة ومفيدة، لا مجرد إجابات تبدو مقنعة.

المجالات:
العلوم، الرياضيات، البرمجة، التعليم، البحث، الكتابة،
الترجمة، التخطيط، تحليل البيانات والصور.

قواعد الدقة:
1. لا تختلق حقائق أو أرقامًا أو دراسات أو أسماء باحثين.
2. لا تختلق روابط أو مراجع أو نتائج بحث.
3. ميّز بين الحقيقة المثبتة، والاستنتاج، والفرضية، والرأي.
4. إذا لم تعرف، أو لم تكف الأدلة، فقل ذلك بوضوح.
5. لا تجعل الإجابة تبدو مؤكدة إذا كانت الأدلة ضعيفة.
6. لا تعتبر تكرار الادعاء في مواقع متعددة دليلًا مستقلًا على صحته.
7. لا تخلط بين الارتباط والسببية.
8. لا تعمم نتائج دراسة واحدة على جميع الحالات.
9. تحقّق من الأرقام والوحدات والتواريخ والأسماء.
10. صحح افتراضات المستخدم إذا كانت خاطئة، مع توضيح السبب.
11. لا توافق المستخدم لمجرد إرضائه.
12. لا تقل إنك بحثت أو اختبرت أو شغّلت شيئًا إلا إذا حدث ذلك فعلًا.

البحث والمصادر:
- عند توفر أداة البحث واستخدامها، اعتمد على ما وجدته فعلًا.
- فضّل المصادر الأصلية والرسمية والأبحاث العلمية المحكمة.
- افحص هل المصدر يدعم الادعاء المحدد، لا مجرد تشابه الموضوع.
- انتبه إلى تاريخ المصدر وإلى احتمال أن تكون المعلومات قديمة.
- لا تعتبر عنوان الصفحة وحده دليلًا على محتواها.
- لا تخترع رابطًا إذا لم تحصل على الرابط الفعلي.
- إذا كانت نتائج البحث غير كافية، اذكر أن التحقق غير مكتمل.
- فرّق بين مصدر قرأته فعلًا ورابط اقترحه الرد ولم يتم فحصه.
- لا تدّعِ أنك فتحت صفحة أو قرأت دراسة إذا لم يحدث ذلك.
- عند تدقيق قائمة ادعاءات، افحص كل ادعاء على حدة.
- صنّف الادعاءات عند الحاجة: صحيح، صحيح جزئيًا،
  خاطئ، غير مدعوم، أو لا تكفي الأدلة للحكم.
- لا تمنح تقييمًا رقميًا دون ذكر معيار التقييم.

الرياضيات:
- حلّ المسألة وتحقق من الناتج والوحدات.
- اشرح الخطوات الضرورية، ولا تخفِ افتراضًا يؤثر في النتيجة.
- استخدم أداة حساب متاحة إذا احتاجت المسألة إلى حساب موثوق.
- لا تدّعِ استخدام أداة حساب إذا لم تستخدمها.

البرمجة:
- افحص الاستيرادات وأنواع البيانات وتدفق التنفيذ والأخطاء المحتملة.
- لا تدّعِ أن الكود اختُبر ما لم تشغّله فعلًا.
- حافظ على الوظائف الموجودة عند تعديل مشروع.
- لا تعرض مفاتيح API أو الأسرار.
- اشرح بوضوح أي ميزة تعتمد على خدمة خارجية.
- ميّز بين الكود المقترح والكود الذي جرى اختباره.

الصور والبيانات:
- استنتج من المعلومات المتاحة فقط.
- وضّح إذا كانت الصورة غير واضحة أو البيانات ناقصة.
- لا تخترع تفاصيل لا تظهر في المدخلات.

أسلوب الإجابة:
- أجب بلغة المستخدم وأسلوبه.
- كن مباشرًا وتجنب الحشو والتكرار.
- قدّم التفاصيل بحسب حاجة السؤال.
- في الأسئلة البسيطة، أجب باختصار.
- في المسائل المعقدة، نظّم النتيجة والأدلة والقيود.
- لا تعرض سلسلة التفكير الداخلية؛ قدّم النتيجة والأسباب المفيدة.

مراجعة قبل الإرسال:
- هل أجبت عن السؤال نفسه؟
- هل توجد معلومة غير مؤكدة صغتها كحقيقة؟
- هل الأرقام والوحدات متسقة؟
- هل المصدر المذكور يدعم الادعاء فعلًا؟
- هل توجد مبالغة أو تناقض؟
- هل أوضحت حدود ما أمكن التحقق منه؟
صحّح ما تستطيع قبل إرسال الإجابة.
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
# 4. زر نسخ الإجابة
# =====================================

def render_copy_button(text, button_id):
    safe_text = json.dumps(
        text,
        ensure_ascii=False
    ).replace("</", "<\\/")

    safe_id = html.escape(str(button_id), quote=True)

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
              cursor: pointer;
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
                  status.textContent =
                    "تعذر النسخ التلقائي؛ انسخ النص يدويًا.";
                }}
              }}
            }});
          </script>
        </body>
        </html>
        """,
        height=48,
        scrolling=False
    )


# =====================================
# 5. النماذج المتاحة
# =====================================

@st.cache_data(ttl=300)
def get_models():
    result = client.models.list()
    return sorted(model.id for model in result.data)


try:
    available_models = get_models()
except Exception as e:
    st.error(f"تعذر جلب النماذج من Groq: {e}")
    st.stop()

chat_available = CHAT_MODEL in available_models
vision_available = VISION_MODEL in available_models


# =====================================
# 6. ذاكرة المحادثة
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_search_used" not in st.session_state:
    st.session_state.last_search_used = False


# =====================================
# 7. حفظ واستعادة المحادثة
# =====================================

with st.sidebar:
    st.header("⚙️ الإعدادات")

    st.caption(f"الدردشة: {CHAT_MODEL}")
    st.caption(f"الصور: {VISION_MODEL}")

    if not chat_available:
        st.warning("نموذج الدردشة غير ظاهر في قائمة حسابك.")

    if not vision_available:
        st.warning("نموذج الصور غير ظاهر في قائمة حسابك.")

    st.divider()
    st.subheader("💾 ذاكرة المحادثة")

    st.caption(
        "السجل محفوظ داخل الجلسة الحالية. "
        "نزّل نسخة إذا أردت الاحتفاظ به."
    )

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
                        and item.get("role") in ["user", "assistant"]
                        and isinstance(item.get("content"), str)
                    ):
                        valid_messages.append({
                            "role": item["role"],
                            "content": item["content"]
                        })

                st.session_state.messages = valid_messages
                st.session_state.last_sources = []
                st.session_state.last_search_used = False

                st.success("تمت استعادة سجل المحادثة.")
                st.rerun()

            except Exception as e:
                st.error(f"تعذرت استعادة الملف: {e}")

    if st.button("مسح المحادثة الحالية", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.last_search_used = False
        st.rerun()

    st.divider()

    enable_web_search = st.toggle(
        "🔎 البحث في الإنترنت",
        value=False
    )

    st.caption(
        "يستخدم البحث أداة Groq المدمجة مع نموذج GPT-OSS. "
        "تظل جودة التحقق معتمدة على النتائج التي يعثر عليها."
    )


# =====================================
# 8. تحويل الصورة
# =====================================

def image_to_data_url(image_file, max_side=1024):
    image_file.seek(0)

    with Image.open(image_file) as original:
        image = original.convert("RGB")
        image.thumbnail((max_side, max_side))

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=80,
            optimize=True
        )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


# =====================================
# 9. تجهيز تاريخ المحادثة
# =====================================

def build_history():
    history = []

    for item in st.session_state.messages[-HISTORY_LIMIT:]:
        if item.get("role") in ["user", "assistant"]:
            history.append({
                "role": item["role"],
                "content": item["content"]
            })

    return history


# =====================================
# 10. استخراج الروابط
# =====================================

def extract_urls(obj, answer=""):
    """
    تستخرج الروابط التي ظهرت في بيانات الاستجابة أو نصها.
    وجود الرابط لا يعني أن محتواه فُحص أو أنه يدعم الادعاء.
    """
    found = []

    def add_url(url, title=""):
        if not isinstance(url, str):
            return

        url = url.strip().rstrip(".,;)]}")

        if not url.startswith(("https://", "http://")):
            return

        found.append({
            "title": str(title or url),
            "url": url
        })

    def walk(value):
        if isinstance(value, dict):
            url = value.get("url")
            title = value.get("title") or value.get("name")

            if isinstance(url, str):
                add_url(url, title)

            for child in value.values():
                walk(child)

        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(obj)

    # التقاط الروابط المكتوبة صراحةً في نص الإجابة.
    for url in re.findall(r"https?://[^\s<>\"']+", answer or ""):
        add_url(url)

    unique = []
    seen = set()

    for source in found:
        if source["url"] not in seen:
            seen.add(source["url"])
            unique.append(source)

    return unique


# =====================================
# 11. الدردشة العادية
# =====================================

def call_chat(user_text):
    if not chat_available:
        raise RuntimeError(
            f"النموذج {CHAT_MODEL} غير متاح في حسابك."
        )

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

    answer = response.choices[0].message.content or ""
    return answer


# =====================================
# 12. البحث والتحقق
# =====================================

def research_with_web(user_text):
    if not chat_available:
        raise RuntimeError(
            f"النموذج {CHAT_MODEL} غير متاح في حسابك."
        )

    research_prompt = SYSTEM_PROMPT + """

وضع البحث والتحقق:
- استخدم أداة البحث المتاحة في هذا الطلب.
- ابحث عن الأدلة المرتبطة بالسؤال، ولا تعتمد على الذاكرة وحدها.
- عند تدقيق ادعاءات، افحص كل ادعاء منفصلًا.
- لا تخلط بين العثور على صفحة وبين التأكد من صحة محتواها.
- لا تذكر مصدرًا على أنه يؤيد الادعاء إذا لم يظهر لك ما يدعم ذلك.
- لا تخترع عناوين أو روابط أو أسماء دراسات.
- أورد الروابط الفعلية فقط عندما تكون متاحة في نتائج البحث.
- إذا لم تتوفر روابط مباشرة، وضّح أن الروابط لم تظهر لك.
- إذا تعارضت المصادر، اذكر التعارض بدل إخفائه.
- إذا كانت الأدلة غير كافية، قل ذلك بوضوح.

رتّب الإجابة حسب الحاجة:
1. النتيجة المختصرة.
2. الأدلة والتفسير.
3. ما هو صحيح جزئيًا أو غير مؤكد.
4. المصادر والروابط التي أمكن تحديدها فعلًا.

لا تدّعِ أن كل مصدر قد فُحص بالكامل إذا لم يحدث ذلك.
"""

    messages = [
        {
            "role": "system",
            "content": research_prompt
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
    answer = message.content or ""

    try:
        response_data = response.model_dump()
    except Exception:
        response_data = {}

    sources = extract_urls(response_data, answer)

    if not answer.strip():
        answer = "لم يرجع البحث إجابة نصية. حاول صياغة السؤال بشكل أوضح."

    if not sources:
        answer += (
            "\n\n**حدود التحقق:** لم أستطع استخراج روابط مباشرة "
            "من الاستجابة. هذا لا يثبت عدم وجود مصادر؛ "
            "لكنه يعني أنني لا أستطيع عرض قائمة روابط موثوقة هنا."
        )

    return answer, sources


# =====================================
# 13. تحليل الصور
# =====================================

def analyze_image(user_text, image_file):
    if not vision_available:
        raise RuntimeError(
            f"النموذج {VISION_MODEL} غير متاح في حسابك."
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
                    or "حلل الصورة بدقة، وميّز بين ما تراه "
                       "وما لا تستطيع التأكد منه."
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
# 14. واجهة المساعد
# =====================================

st.title("🤖 مساعد الذكاء الاصطناعي")

st.caption(
    "دردشة ذكية • تحليل الصور • بحث عند توفره • سجل محادثات"
)

if not st.session_state.messages:
    st.info(
        "ابدأ بسؤالك. فعّل البحث عندما تحتاج معلومات حديثة "
        "أو تريد تدقيق ادعاءات ومصادر."
    )


# عرض سجل المحادثة
for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_copy_button(
                message["content"],
                f"history_{index}"
            )


# عرض الروابط التي ظهرت في الاستجابة
if st.session_state.last_sources:
    with st.expander("🔗 روابط ظهرت في الاستجابة"):
        st.warning(
            "وجود الرابط هنا لا يعني أن محتواه فُحص بشكل مستقل "
            "أو أنه يثبت كل ادعاءات الإجابة."
        )

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
                    used_search = False

                elif enable_web_search:
                    answer, sources = research_with_web(question)
                    used_search = True

                else:
                    answer = call_chat(question)
                    sources = []
                    used_search = False

            st.session_state.messages.append({
                "role": "user",
                "content": saved_question
            })

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            st.session_state.last_sources = sources
            st.session_state.last_search_used = used_search

            st.rerun()

        except Exception as e:
            st.error(f"حدث خطأ: {e}")
            st.info(
                "إذا فشل البحث، تحقق من إتاحة الأداة وحدود حساب Groq. "
                "وإذا فشل تحليل الصورة، تحقق من دعم النموذج للصورة."
            )

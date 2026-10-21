
import os
import io
import json
import base64
import html

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


# =====================================
# 2. تعليمات الذكاء الشامل
# =====================================

SYSTEM_PROMPT = """
أنت مساعد ذكاء اصطناعي عام متعدد المجالات.
هدفك فهم طلب المستخدم وحل مشكلاته بدقة ووضوح.

مجالاتك:
- العلوم والرياضيات والتحليل المنطقي.
- البرمجة وتصحيح الأخطاء وتصميم التطبيقات.
- التعليم وشرح المفاهيم من الأساسي إلى المتقدم.
- البحث والتحقق من الحقائق ومقارنة المصادر.
- الكتابة والتلخيص والترجمة والتخطيط.
- تحليل الصور والبيانات عندما تكون الأدوات متاحة.

الفهم والاستدلال:
- افهم مقصد المستخدم وسياق السؤال.
- حلل المسائل المعقدة إلى خطوات واضحة.
- افحص الافتراضات والاحتمالات البديلة.
- تحقق من النتائج والحسابات عند الإمكان.
- إذا كان الغموض يؤثر في الإجابة، اسأل سؤالًا توضيحيًا.
- لا تعرض التخمين على أنه حقيقة.

الدقة والبحث:
- لا تختلق حقائق أو أرقامًا أو مصادر أو روابط.
- لا تقل إنك بحثت في الإنترنت ما لم تستخدم أداة بحث فعلية.
- استخدم المصادر المتاحة فعلًا، وتحقق أنها تدعم الادعاء نفسه.
- فضّل المصادر الأصلية والرسمية والأبحاث المحكمة.
- ميّز بين الحقيقة المدعومة والاستنتاج والفرضية.
- لا تعمم نتيجة دراسة محدودة على جميع الحالات.
- اذكر القيود والشكوك المهمة.
- إذا لم تتمكن من التحقق، صرّح بذلك.
- لا تدّعِ وجود إجماع علمي دون دليل كافٍ.
- لا تخترع روابط للمراجع ولا تستخدم روابط تخمينية.
- لا تجعل غرابة المعلومة أهم من صحتها.

الرياضيات والعلوم:
- اشرح خطوات الحل الضرورية.
- تحقق من العمليات الحسابية والوحدات والشروط.
- ميّز بين الارتباط والسببية.
- لا تصف فرضية بأنها حقيقة مثبتة.
- اذكر الظروف التي قد تغيّر النتيجة.

البرمجة والتقنية:
- اكتب كودًا واضحًا ومتكاملًا قدر الإمكان.
- راجع الاستيرادات والمتغيرات والمدخلات والأخطاء المحتملة.
- حافظ على وظائف المشروع السابقة عند تعديل الكود.
- وضّح مكان وضع الكود وطريقة تشغيله.
- لا تدّعِ تشغيل الكود أو اختباره ما لم يحدث ذلك فعلًا.
- لا تكشف مفاتيح API أو كلمات المرور السرية.
- لا تدّعِ أن ميزة أو أداة تعمل قبل التأكد من توفرها.
- إذا لم تتوفر أداة مطلوبة، اشرح القيد بوضوح.

التعليم:
- اشرح بحسب مستوى المستخدم.
- ابدأ من الأساسيات عند الحاجة.
- استخدم أمثلة وتمارين عملية.
- إذا طلب المستخدم الاختصار، أعطه الزبدة دون حشو.

الكتابة والتحليل والتخطيط:
- تكيّف مع لغة المستخدم وأسلوبه.
- في المقارنات، استخدم معايير واضحة.
- لا تخترع بيانات ناقصة.
- افصل الحقائق عن الآراء.
- راعِ القيود والوقت والميزانية المذكورة.

الصور والبيانات:
- صف ما تستطيع رؤيته فقط.
- وضّح إذا كانت الصورة غير واضحة.
- لا تدّعِ قراءة تفاصيل لا تستطيع تمييزها.
- تحقق من البيانات قبل استنتاج النتائج.

مراجعة الإجابة:
قبل الإرسال، راجع دقة المعلومات، والتناقضات،
والمبالغة في الثقة، والأرقام والمصادر.
صحح الأخطاء التي تستطيع اكتشافها.

لا تعرض سلسلة التفكير الداخلية الخاصة.
قدّم النتيجة والأسباب والخطوات المفيدة بدلًا منها.

أسلوب التواصل:
- استخدم لغة المستخدم.
- كن صريحًا ومباشرًا.
- تجنب الحشو والتكرار والمجاملات المبالغ فيها.
- لا توافق المستخدم لمجرد إرضائه.
- الدقة أولًا، ثم الفائدة والوضوح.
"""


# =====================================
# 3. التصميم
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
# 4. مفتاح Groq
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
# 5. زر نسخ الإجابة
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
# 6. فحص النماذج
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
# 7. ذاكرة المحادثة
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# =====================================
# 8. حفظ واستعادة المحادثة
# =====================================

with st.sidebar:
    st.header("⚙️ الإعدادات")

    st.caption(f"الدردشة: {CHAT_MODEL}")
    st.caption(f"الصور: {VISION_MODEL}")

    if not chat_available:
        st.warning("نموذج الدردشة غير ظاهر في حساب Groq.")

    if not vision_available:
        st.warning("نموذج الصور غير ظاهر في حساب Groq.")

    st.divider()
    st.subheader("💾 ذاكرة المحادثة")

    st.caption(
        "المحادثات محفوظة داخل الجلسة الحالية. "
        "استخدم التنزيل للاحتفاظ بنسخة."
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

                st.success("تمت استعادة سجل المحادثة.")
                st.rerun()

            except Exception as e:
                st.error(f"تعذرت استعادة الملف: {e}")

    if st.button("مسح المحادثة الحالية", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()

    st.divider()

    enable_web_search = st.toggle(
        "🔎 البحث في الإنترنت",
        value=False
    )

    st.caption(
        "قد يتطلب البحث دعم الأداة من Groq للنموذج وحسابك."
    )


# =====================================
# 9. تحويل الصورة
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
# 10. تجهيز تاريخ المحادثة
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
# 11. استخراج الروابط
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
# 12. الدردشة العادية
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
# 13. البحث في الإنترنت
# =====================================

def research_with_web(user_text):
    if not chat_available:
        raise RuntimeError(
            f"نموذج الدردشة {CHAT_MODEL} غير متاح في حسابك."
        )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + """
عند البحث:
- استخدم نتائج أداة البحث الفعلية فقط.
- تحقق أن المصدر يدعم الادعاء المحدد.
- فضّل المصادر الأصلية والموثوقة والحديثة.
- لا تخترع روابط أو مراجع.
- إذا لم تصل نتائج بحث فعلية، صرّح بأن التحقق لم يكتمل.
"""
        }
    ]

    messages.extend(build_history())

    messages.append({
        "role": "user",
        "content": user_text
    })

    # قد تختلف إتاحة browser_search بحسب دعم Groq الحالي.
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

    try:
        response_data = response.model_dump()
    except Exception:
        response_data = {}

    sources = extract_urls(response_data)
    answer = message.content or ""

    if not sources:
        answer += (
            "\n\n**تنبيه:** لم أتمكن من استخراج روابط مصادر مباشرة "
            "من استجابة البحث؛ تحقق من المراجع قبل الاعتماد على الإجابة."
        )

    return answer, sources


# =====================================
# 14. تحليل الصور
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
# 15. واجهة المساعد
# =====================================

st.title("🤖 مساعد الذكاء الاصطناعي")

st.caption(
    "دردشة ذكية • تحليل الصور • بحث عند توفره • سجل محادثات"
)

if not st.session_state.messages:
    st.info(
        "ابدأ بسؤالك. فعّل البحث إذا احتجت معلومات حديثة ومصادر."
    )


# عرض سجل المحادثة مع زر نسخ تحت كل إجابة
for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_copy_button(
                message["content"],
                f"history_{index}"
            )


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
                "إذا ظهر الخطأ عند البحث، فقد تكون أداة البحث غير "
                "مدعومة بهذه الصيغة. تحقق من توثيق Groq الحالي."
            )

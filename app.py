import streamlit as st
import google.generativeai as genai
from PIL import Image

# 1. إعداد واجهة التطبيق وتصميم الـ CSS لتكون شبيهة بواجهة الشات الحديثة
st.set_page_config(
    page_title="الذكاء الموسوعي الخارق - أنس",
    page_icon="⚡",
    layout="centered"
)

# تخصيص التصميم والألوان لتكون مرتبة وعصرية
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTextArea textarea {
        background-color: #161b22;
        color: #ffffff;
        border-radius: 12px;
        border: 1px solid #30363d;
    }
    .stButton button {
        background-color: #238636;
        color: white;
        border-radius: 10px;
        font-weight: bold;
        border: none;
        width: 100%;
        padding: 10px;
    }
    .stButton button:hover {
        background-color: #2ea043;
    }
    </style>
""", unsafe_allow_html=True)

# عنوان التطبيق الأنيق
st.title("⚡ الذكاء الموسوعي الخارق")
st.caption("مرحباً بك يا أنس، منصتك الخاصة للتحليل والبحث الفائق.")

# جلب المفتاح بأمان من إعدادات Streamlit Secrets
MY_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# خانة رفع الصور بتصميم مبسط
uploaded_file = st.file_uploader("ارفق صورة للتحليل البصري (اختياري):", type=["jpg", "jpeg", "png"])

# خانة السؤال أو المشكلة
user_question = st.text_area("اطرح سؤالك أو معضلتك هنا:", placeholder="اكتب ما تبحث عنه...")

if st.button("إرسال وتحليل 🚀"):
    if not MY_API_KEY:
        st.error("لم يتم العثور على المفتاح في إعدادات Secrets في Streamlit!")
    else:
        try:
            genai.configure(api_key=MY_API_KEY)
            
            system_instruction = (
                "أنت خبير موسوعي وعالم ذكاء اصطناعي فائق الذكاء، ملم بجميع العلوم والمعارف البشرية. "
                "أجب عن أي سؤال يطرحه المستخدم (أنس) بعمق، ودقة علمية عالية، ومعلومات دقيقة وغنية دون إجابات سطحيّة."
            )
            
            model = genai.GenerativeModel(
                model_name='gemini-2.5-flash',
                system_instruction=system_instruction
            )
            
            with st.spinner("جاري التفكير وصياغة الإجابة... 🧠"):
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    if user_question:
                        response = model.generate_content([user_question, image])
                    else:
                        response = model.generate_content(["قم بتحليل هذه الصورة بدقة وتفصيل:", image])
                else:
                    if not user_question:
                        st.warning("الرجاء كتابة سؤال أو رفع صورة للبدء.")
                        st.stop()
                    else:
                        response = model.generate_content(user_question)
                        
                st.markdown("### 💡 الإجابة:")
                st.markdown(response.text)
                
        except Exception as e:
            st.error(f"حدث خطأ: {e}")

import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="الذكاء الموسوعي الخارق - أنس",
    page_icon="⚡",
    layout="centered"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stTextArea textarea { background-color: #161b22; color: #ffffff; border-radius: 12px; border: 1px solid #30363d; }
    .stButton button { background-color: #238636; color: white; border-radius: 10px; font-weight: bold; border: none; width: 100%; padding: 10px; }
    .stButton button:hover { background-color: #2ea043; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ الذكاء الموسوعي الخارق (Groq Engine)")
st.caption("مرحباً بك يا أنس، الأداء السريع والدقيق.")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

user_question = st.text_area("اطرح سؤالك أو معضلتك هنا:", placeholder="اكتب ما تبحث عنه...")

if st.button("إرسال وتحليل 🚀"):
    if not GROQ_API_KEY:
        st.error("لم يتم العثور على مفتاح Groq في إعدادات Secrets في Streamlit!")
    else:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            
            if not user_question:
                st.warning("الرجاء كتابة سؤال للبدء.")
                st.stop()
            
            with st.spinner("جاري المعالجة... ⚡"):
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "أنت خبير موسوعي وعالم ذكاء اصطناعي دقيق، تقدم معلومات موثوقة بعيداً عن التهويل أو الخلط غير العلمي."
                        },
                        {
                            "role": "user",
                            "content": user_question,
                        }
                    ],
                    model="llama-3.3-70b-versatile",
                )
                
                response_text = chat_completion.choices[0].message.content
                
                st.markdown("### 💡 الإجابة:")
                st.markdown(response_text)
                
        except Exception as e:
            st.error(f"حدث خطأ التقنية: {e}")

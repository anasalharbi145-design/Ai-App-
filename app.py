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
st.caption("مرحباً بك يا أنس، السرعة القصوى والأداء الفائق.")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

user_question = st.text_area("اطرح سؤالك أو معضلتك هنا:", placeholder="اكتب ما تبحث عنه...")

if st.button("إرسال وتحليل صاروخي 🚀"):
    if not GROQ_API_KEY:
        st.error("لم يتم العثور على مفتاح Groq في إعدادات Secrets في Streamlit!")
    else:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            
            if not user_question:
                st.warning("الرجاء كتابة سؤال للبدء.")
                st.stop()
            
            with st.spinner("جاري المعالجة بسرعة البرق... ⚡"):
                # قائمة بالنماذج المتاحة في Groq بالترتيب
                candidate_models = [
                    "llama-3.1-8b-instant",
                    "llama-3.3-70b-versatile",
                    "llama3-70b-8192",
                    "mixtral-8x7b-32768"
                ]
                
                chat_completion = None
                last_error = None
                
                # المحاولة عبر النماذج المتاحة تلقائياً
                for model_name in candidate_models:
                    try:
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "system",
                                    "content": "أنت خبير موسوعي وعالم ذكاء اصطناعي فائق الذكاء، ملم بجميع العلوم والمعارف. أجب بعمق ودقة علمية عالية."
                                },
                                {
                                    "role": "user",
                                    "content": user_question,
                                }
                            ],
                            model=model_name,
                        )
                        if chat_completion:
                            break
                    except Exception as err:
                        last_error = err
                        continue
                
                if chat_completion:
                    response_text = chat_completion.choices[0].message.content
                    st.markdown("### 💡 الإجابة:")
                    st.markdown(response_text)
                else:
                    st.error(f"تعذر الاتصال بجميع النماذج. الخطأ: {last_error}")
                
        except Exception as e:
            st.error(f"حدث خطأ: {e}")

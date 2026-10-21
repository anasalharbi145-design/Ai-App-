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
st.caption("مرحباً بك يا أنس، جلب الموديلات المتاحة تلقائياً في حسابك.")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

selected_model = "llama-3.1-8b-instant"

if not GROQ_API_KEY:
    st.error("لم يتم العثور على مفتاح Groq في إعدادات Secrets!")
else:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        # جلب النماذج المتاحة فعلياً في المفتاح واستبعاد النماذج غير الخاصة بالمحادثة
        models_response = client.models.list()
        available_models = [
            m.id for m in models_response.data 
            if "guard" not in m.id.lower() and "whisper" not in m.id.lower() and "embed" not in m.id.lower()
        ]
        
        if available_models:
            # قائمة منسدلة تتيح لك اختيار الموديل الشغال في حسابك بكل حرية
            selected_model = st.selectbox("اختر الموديل المتوافق مع مفتاحك:", available_models)
        else:
            st.warning("لم يتم العثور على نماذج تلقائية، سيتم استخدام الموديل الافتراضي.")
            
    except Exception as e:
        st.info("جاري استخدام الإعداد الافتراضي للموديل...")

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
                    model=selected_model,
                )
                
                response_text = chat_completion.choices[0].message.content
                
                st.markdown("### 💡 الإجابة:")
                st.markdown(response_text)
                
        except Exception as e:
            st.error(f"حدث خطأ: {e}")

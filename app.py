import streamlit as st
from groq import Groq
import base64

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
st.caption("مرحباً بك يا أنس، النظام الذكي بدون أخطاء موديلات.")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

uploaded_file = st.file_uploader("ارفق صورة للتحليل البصري (اختياري):", type=["jpg", "jpeg", "png"])
user_question = st.text_area("اطرح سؤالك أو معضلتك هنا:", placeholder="اكتب ما تبحث عنه...")

if st.button("إرسال وتحليل صاروخي 🚀"):
    if not GROQ_API_KEY:
        st.error("لم يتم العثور على مفتاح Groq في إعدادات Secrets!")
    else:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            
            if not user_question and not uploaded_file:
                st.warning("الرجاء كتابة سؤال أو رفع صورة للبدء.")
                st.stop()
            
            with st.spinner("جاري جلب النموذج المتاح والمعالجة... ⚡"):
                # جلب كل الموديلات المتاحة في حسابك المباشر
                all_models = [m.id for m in client.models.list().data]
                
                if uploaded_file is not None:
                    # اختيار موديل يدعم الصور من الحساب
                    vision_models = [m for m in all_models if "vision" in m.lower() or "llava" in m.lower()]
                    model_to_use = vision_models[0] if vision_models else "llama-3.2-11b-vision-preview"
                    
                    bytes_data = uploaded_file.getvalue()
                    base64_image = base64.b64encode(bytes_data).decode('utf-8')
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    mime_type = f"image/{'jpeg' if file_extension in ['jpg', 'jpeg'] else file_extension}"
                    
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_question if user_question else "اشرح هذه الصورة بالتفصيل."},
                                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                            ]
                        }
                    ]
                else:
                    # اختيار أول موديل محادثة نصوص فعال في حسابك تلقائياً
                    text_models = [m for m in all_models if "guard" not in m.lower() and "whisper" not in m.lower() and "vision" not in m.lower()]
                    model_to_use = text_models[0] if text_models else all_models[0]
                    
                    messages = [
                        {"role": "system", "content": "أنت خبير موسوعي وعالم ذكاء اصطناعي فائق الذكاء، أجب بعمق ودقة علمية عالية."},
                        {"role": "user", "content": user_question}
                    ]

                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model=model_to_use,
                )
                
                response_text = chat_completion.choices[0].message.content
                st.markdown("### 💡 الإجابة:")
                st.markdown(response_text)
                
        except Exception as e:
            st.error(f"حدث خطأ: {e}")

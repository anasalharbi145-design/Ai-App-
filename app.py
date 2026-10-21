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
st.caption("مرحباً بك يا أنس، استنتاج أقوى نموذج متاح مع زر النسخ المباشر.")

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
            
            with st.spinner("جاري استنتاج واختبار أقوى نموذج متاح في حسابك... ⚡"):
                models_response = client.models.list()
                all_model_ids = [m.id for m in models_response.data]
                
                response_text = None
                used_model_name = ""
                last_error = None
                
                if uploaded_file is not None:
                    # فلترة وترتيب نماذج الصور (تفضيل الأقوى والأحدث)
                    vision_candidates = [m for m in all_model_ids if any(k in m.lower() for k in ["vision", "llava", "90b", "11b"])]
                    vision_candidates.sort(key=lambda x: ("90b" in x.lower(), "vision" in x.lower()), reverse=True)
                    if not vision_candidates:
                        vision_candidates = all_model_ids
                        
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
                    
                    for m_id in vision_candidates:
                        try:
                            chat_completion = client.chat.completions.create(
                                messages=messages,
                                model=m_id,
                            )
                            response_text = chat_completion.choices[0].message.content
                            used_model_name = m_id
                            break
                        except Exception as err:
                            last_error = err
                            continue
                else:
                    # فلترة نماذج النصوص واستبعاد الحماية
                    text_candidates = [
                        m for m in all_model_ids 
                        if "guard" not in m.lower() 
                        and "whisper" not in m.lower() 
                        and "embed" not in m.lower()
                        and "canopylabs" not in m.lower()
                    ]
                    # ترتيب ذكي: جعل النماذج الكبرى والقوية (70b / versatile) في المقدمة لتكون الخيار الأول دائماً
                    text_candidates.sort(key=lambda x: ("70b" in x.lower(), "versatile" in x.lower(), "instant" in x.lower()), reverse=True)
                    
                    if not text_candidates:
                        text_candidates = all_model_ids
                        
                    messages = [
                        {"role": "system", "content": "أنت خبير موسوعي وعالم ذكاء اصطناعي فائق الذكاء، أجب بعمق ودقة علمية عالية."},
                        {"role": "user", "content": user_question}
                    ]
                    
                    for m_id in text_candidates:
                        try:
                            chat_completion = client.chat.completions.create(
                                messages=messages,
                                model=m_id,
                            )
                            response_text = chat_completion.choices[0].message.content
                            used_model_name = m_id
                            break
                        except Exception as err:
                            last_error = err
                            continue
                            
                if response_text:
                    st.success(f"🚀 تم اختيار التشغيل بنجاح عبر أقوى نموذج متاح: **{used_model_name}**")
                    
                    st.markdown("### 💡 الإجابة:")
                    st.markdown(response_text)
                    
                    # عرض زر نسخ النص المباشر في نهاية الإجابة
                    st.markdown("---")
                    st.markdown("📋 **نسخ النص بنقرة واحدة:**")
                    st.code(response_text, language="markdown")
                else:
                    st.error(f"عذراً، لم ينجح أي نموذج في حسابك. الخطأ: {last_error}")
                
        except Exception as e:
            st.error(f"حدث خطأ عام: {e}")

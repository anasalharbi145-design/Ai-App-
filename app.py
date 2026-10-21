import streamlit as st
import google.generativeai as genai
from PIL import Image

st.title("🤖 الذكاء الاصطناعي الخاص بـ أنس")
st.write("أهلاً بك يا أنس! تطبيقك الخاص الآن يدعم النصوص والصور معاً بذكاء Gemini.")

# خانة إدخال المفتاح
api_key = st.text_input("أدخل الـ API Key الخاص بك:", type="password")

# إضافة زر لرفع الصور
uploaded_file = st.file_uploader("اختر صورة للإرسال (اختياري):", type=["jpg", "jpeg", "png"])

# خانة السؤال
user_question = st.text_area("اطرح سؤالك أو اطلب أي شيء عن الصورة أو النص:")

if st.button("إرسال 🚀"):
    if not api_key:
        st.error("يا طويل العمر، لا تنسى تحط الـ API Key أولاً!")
    else:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            with st.spinner("جاري المعالجة والتفكير..."):
                # التحقق إذا فيه صورة مرفوعة أو لا
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    if user_question:
                        response = model.generate_content([user_question, image])
                    else:
                        response = model.generate_content(["صف لي هذه الصورة بالتفصيل:", image])
                else:
                    if not user_question:
                        st.warning("الرجاء كتابة سؤال أو رفع صورة على الأقل.")
                        st.stop()
                    else:
                        response = model.generate_content(user_question)
                        
                st.success("تمت الإجابة بنجاح:")
                st.write(response.text)
                
        except Exception as e:
            st.error(f"تأكد أن المفتاح صحيح، خطأ: {e}")

import streamlit as st
import google.generativeai as genai

st.title("🤖 الذكاء الاصطناعي الخاص بـ أنس")
st.write("أهلاً بك! هذا تطبيقك الخاص مدعوم بذكاء Google الحقيقي.")

# خانة إدخال المفتاح
api_key = st.text_input("أدخل الـ API Key الخاص بك:", type="password")

# خانة السؤال
user_question = st.text_area("اطرح سؤالك أو اطلب أي شيء من الذكاء الاصطناعي:")

if st.button("إرسال 🚀"):
    if not api_key:
        st.error("يا طويل العمر، لا تنسى تحط الـ API Key أولاً!")
    elif not user_question:
        st.warning("اكتب سؤالك في المربع فوق عشان أقدر أساعدك.")
    else:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            with st.spinner("جاري التفكير وصياغة الإجابة..."):
                response = model.generate_content(user_question)
                st.success("تمت الإجابة بنجاح:")
                st.write(response.text)
        except Exception as e:
            st.error(f"تأكد أن المفتاح صحيح، خطأ: {e}")

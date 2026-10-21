import streamlit as st
import google.generativeai as genai
from PIL import Image

# إعداد صفحة التطبيق
st.set_page_config(page_title="الذكاء الاصطناعي الموسوعي - أنس", page_icon="🌐", layout="centered")

st.title("🌐 الذكاء الاصطناعي الموسوعي الخارق بـ أنس")
st.write("أهلاً بك يا أنس! تطبيقك المطور جاهز للتحليل العميق والإجابة عن أي مجال في الحياة (علوم، تاريخ، تقنية، ثقافة، وتحليل صور).")

# جلب المفتاح بأمان من إعدادات Streamlit Secrets
MY_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# خانة رفع الصور
uploaded_file = st.file_uploader("اختر صورة للتحليل العميق (اختياري):", type=["jpg", "jpeg", "png"])

# خانة السؤال النصي
user_question = st.text_area("اطرح سؤالك أو موضوعك في أي مجال:")

if st.button("إطلاق التحليل الخارق 🚀"):
    if not MY_API_KEY:
        st.error("لم يتم العثور على المفتاح في إعدادات Secrets في Streamlit!")
    else:
        try:
            # تهيئة المفتاح تلقائياً
            genai.configure(api_key=MY_API_KEY)
            
            # توجيهات نظامية موسوعية شاملة لكل العلوم والمجالات
            system_instruction = (
                "أنت خبير ومستشار موسوعي فائق الذكاء، عالم بكل المجالات والتخصصات (العلوم الطبيعية، "
                "التاريخ، الفلسفة، الأدب، الاقتصاد، البرمجة والتكنولوجيا، والتحليل العام). "
                "عندما يسألك المستخدم (أنس) عن أي موضوع، قم بتحليله بعمق شديد وبشكل شمول ومنهجي ورائع، "
                "وقدم معلومات غنية ونادرة ودقيقة دون الاقتصار على مجال واحد، مع تجنب الإجابات السطحية نهائياً."
            )
            
            # استخدام أحدث نموذج
            model = genai.GenerativeModel(
                model_name='gemini-3.8-flash',
                system_instruction=system_instruction
            )
            
            with st.spinner("جاري التفكير العميق والتحليل الموسوعي... 🧠"):
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    if user_question:
                        response = model.generate_content([user_question, image])
                    else:
                        response = model.generate_content(["قم بتحليل هذه الصورة بدقة وتفصيل عميق واستخرج منها كل التفاصيل والمعلومات الممكنة:", image])
                else:
                    if not user_question:
                        st.warning("الرجاء كتابة سؤال أو رفع صورة للبدء.")
                        st.stop()
                    else:
                        response = model.generate_content(user_question)
                        
                st.success("تم إعداد الإجابة الموسوعية بنجاح:")
                st.markdown(response.text)
                
        except Exception as e:
            st.error(f"حدث خطأ، تأكد من صحة المفتاح. التفاصيل: {e}")

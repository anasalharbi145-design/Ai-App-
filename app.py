import streamlit as st
import google.generativeai as genai
from PIL import Image

# إعداد صفحة التطبيق
st.set_page_config(page_title="الذكاء الاصطناعي الخارق - أنس", page_icon="⚡", layout="centered")

st.title("⚡ الذكاء الاصطناعي الخارق الخاص بـ أنس")
st.write("أهلاً بك يا أنس! تطبيقك شغال ومربوط بالمفتاح تلقائياً وجاهز للتحليل الفائق.")

# جلب المفتاح بأمان من إعدادات Streamlit Secrets
MY_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# خانة رفع الصور
uploaded_file = st.file_uploader("اختر صورة للتحليل العميق (اختياري):", type=["jpg", "jpeg", "png"])

# خانة السؤال النصي
user_question = st.text_area("اطرح سؤالك أو معضلتك البرمجية/العلمية:")

if st.button("إطلاق التحليل الخارق 🚀"):
    if not MY_API_KEY:
        st.error("لم يتم العثور على المفتاح في إعدادات Secrets في Streamlit!")
    else:
        try:
            # تهيئة المفتاح تلقائياً
            genai.configure(api_key=MY_API_KEY)
            
            # توجيهات النظام الاحترافية
            system_instruction = (
                "أنت خبير تقني وعالم ذكاء اصطناعي عبقري ومحترف جداً. "
                "أجب عن أسئلة المستخدم (أنس) بعمق شديد، دقة متناهية، وبشرح مفصل وواضح ومنهجي. "
                "قم بتحليل كل زاوية من السؤال بدقة قبل إعطاء الإجابة النهائية، وتجنب الإجابات السطحية."
            )
            
            # استخدام أحدث نموذج كما طلبت جوجل في رسالة الخطأ
            model = genai.GenerativeModel(
                model_name='gemini-3.8-flash',
                system_instruction=system_instruction
            )
            
            with st.spinner("جاري التفكير العميق والتحليل الفائق... 🧠"):
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    if user_question:
                        response = model.generate_content([user_question, image])
                    else:
                        response = model.generate_content(["قم بتحليل هذه الصورة بدقة وتفصيل عميق واستخرج منها كل المعلومات الممكنة:", image])
                else:
                    if not user_question:
                        st.warning("الرجاء كتابة سؤال أو رفع صورة للبدء.")
                        st.stop()
                    else:
                        response = model.generate_content(user_question)
                        
                st.success("تم إعداد الإجابة الاحترافية بنجاح:")
                st.markdown(response.text)
                
        except Exception as e:
            st.error(f"حدث خطأ، تأكد من صحة المفتاح. التفاصيل: {e}")

import streamlit as st
import google.generativeai as genai
from PIL import Image

# إعداد صفحة التطبيق لتكون واسعة ومرتبة
st.set_page_config(page_title="الذكاء الاصطناعي الخارق - أنس", page_icon="⚡", layout="centered")

st.title("⚡ الذكاء الاصطناعي الخارق الخاص بـ أنس")
st.write("أهلاً بك يا أنس! تطبيقك المطور الآن يعمل بأحدث وأقوى النماذج وبمستوى تفكير تحليلي فائق.")

# الشريط الجانبي للإعدادات المفتاحية
with st.sidebar:
    st.header("إعدادات النظام الخارق")
    api_key = st.text_input("أدخل الـ API Key الخاص بك:", type="password")
    st.info("💡 نصيحة: اكتب سؤالك أو ارفع صورتك بدقة، وسيقوم النظام بالتحليل العميق الفوري.")

# خانة رفع الصور
uploaded_file = st.file_uploader("اختر صورة للتحليل العميق (اختياري):", type=["jpg", "jpeg", "png"])

# خانة السؤال النصي
user_question = st.text_area("اطرح سؤالك أو معضلتك البرمجية/العلمية:")

if st.button("إطلاق التحليل الخارق 🚀"):
    if not api_key:
        st.error("يا طويل العمر، لا تنسى تحط الـ API Key في القائمة الجانبية أولاً!")
    else:
        try:
            # ربط المفتاح بالذكاء الاصطناعي
            genai.configure(api_key=api_key)
            
            # تهيئة النموذج مع توجيهات نظامية (System Instruction) تخليه دقيق وخارق
            system_instruction = (
                "أنت خبير تقني وعالم ذكاء اصطناعي عبقري ومحترف جداً. "
                "أجب عن أسئلة المستخدم (أنس) بعمق شديد، دقة متناهية، وبشرح مفصل وواضح ومنهجي. "
                "قم بتحليل كل زاوية من السؤال بدقة قبل إعطاء الإجابة النهائية، وتجنب الإجابات السطحية."
            )
            
            # استخدام موديل فلاش الأحدث والأسرع
            model = genai.GenerativeModel(
                model_name='gemini-2.5-flash',
                system_instruction=system_instruction
            )
            
            with st.spinner("جاري التفكير العميق والتحليل الفائق... 🧠"):
                # معالجة الصور أو النصوص بذكاء متقدم
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

import streamlit as st
import google.generativeai as genai
from PIL import Image

# إعداد واجهة التطبيق
st.set_page_config(page_title="الذكاء الموسوعي الخارق - أنس", page_icon="🧠", layout="centered")

st.title("🧠 الذكاء الموسوعي الخارق الخاص بـ أنس")
st.write("أهلاً بك يا أنس! تطبيقك يعمل الآن بكفاءة مطلقة وشاملة لكل مجالات المعرفة (علوم، تاريخ، برمجة، منطق، وتحليل صور).")

# جلب المفتاح بأمان من إعدادات Streamlit Secrets
MY_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# خانة رفع الصور للتحليل البصري
uploaded_file = st.file_uploader("اختر صورة للتحليل العميق (اختياري):", type=["jpg", "jpeg", "png"])

# خانة السؤال أو المشكلة
user_question = st.text_area("اطرح أي سؤال، معضلة، أو فكرة في بالك لأقوم بتحليلها:")

if st.button("إطلاق التحليل والبحث العميق 🚀"):
    if not MY_API_KEY:
        st.error("لم يتم العثور على المفتاح في إعدادات Secrets في Streamlit!")
    else:
        try:
            # تهيئة مفتاح الـ API
            genai.configure(api_key=MY_API_KEY)
            
            # توجيهات النظام (System Instructions) لتشجيع التفكير العميق والدقيق والموسوعي
            system_instruction = (
                "أنت خبير موسوعي وعالم عبقري فائق الذكاء، ملم بجميع العلوم والمعارف البشرية "
                "(من فيزياء وفلك وتاريخ وفلسفة وإدارة وحتى البرمجة وتقنية المعلومات). "
                "عندما يجيب المستخدم (أنس) عن أي سؤال، اتبع منهجية صارمة: "
                "1. دقق في الحقائق العلمية وتجنب المبالغات أو التعميمات غير الموثقة. "
                "2. قدم إجابات غنية ونادرة، واشرح الأسباب والخلفيات بعمق شديد وبدون إجابات سطحيّة. "
                "3. إذا كان هناك فرضيات، ميز بوضوح بين النظريات المثبتة والفرضيات العلمية."
            )
            
            # استخدام أحدث نموذج ذكي
            model = genai.GenerativeModel(
                model_name='gemini-3.8-flash',
                system_instruction=system_instruction
            )
            
            with st.spinner("جاري التفكير العميق والتمحيص العلمي... 🧠"):
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
                        
                st.success("النتيجة والتحليل الموسوعي الدقيق:")
                st.markdown(response.text)
                
        except Exception as e:
            st.error(f"حدث خطأ أثناء الاتصال. التفاصيل: {e}")

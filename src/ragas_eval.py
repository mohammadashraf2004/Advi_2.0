import os
import requests
import time
import pandas as pd
import asyncio
from datasets import Dataset
from dotenv import load_dotenv

# 🌟 التعديلات الجديدة للـ Imports عشان التحذيرات تختفي والـ Objects تشتغل
from ragas import evaluate
from ragas.metrics import AnswerRelevancy, Faithfulness, ContextPrecision, ContextRecall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# تحميل المفاتيح
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY is missing in your .env file!")
    exit(1)

# إعدادات الـ API بتاعك
API_URL = "http://127.0.0.1:8000/api/v1/nlp/index/answer/1"

# 1. تعريف موديلات التقييم (Ragas Judges)
evaluator_llm = ChatOpenAI(model="gpt-4o-mini")
evaluator_embeddings = OpenAIEmbeddings()

# 2. حزمة الاختبار (Test Suite)
# 2. حزمة الاختبار الشاملة (20 سؤال)
TEST_CASES = [
    {"question": "طالب معدله التراكمي ظل أقل من 2.0 لمدة 3 سنوات متتالية (6 فصول)، ما هو وضعه القانوني بالكلية؟", 
     "ground_truth": "يفصل الطالب من الكلية إذا ظل معدله التراكمي أقل من 2.0 لمدة 6 فصول دراسية متتالية."},
    
    {"question": "إذا تغيبت عن المحاضرات بنسبة تجاوزت ربع عدد الساعات المقررة (25%) دون عذر، هل يحق لي دخول الامتحان النهائي؟", 
     "ground_truth": "يُحرم الطالب من دخول الامتحان النهائي في المقرر إذا تجاوزت نسبة غيابه 25% بدون عذر مقبول."},
    
    {"question": "أرغب في رفع معدلي التراكمي بإعادة دراسة بعض المواد التي نجحت فيها بتقدير منخفض، ما هو الحد الأقصى لعدد المواد المسموح لي بتحسينها؟", 
     "ground_truth": "يُسمح للطالب بتحسين تقديره في عدد محدود من المواد (حسب اللائحة غالباً 5 مواد أو ما يرفعه لـ 2.0) مع احتساب التقدير الأعلى."},
    
    {"question": "إذا قمت بالانسحاب الرسمي من مقرر في الأسبوع الثامن، ما هو الرمز الذي سيظهر بجانب المادة في بيان الدرجات؟", 
     "ground_truth": "يظهر الرمز (W) بجانب المقرر عند الانسحاب الرسمي خلال الفترة المحددة (الأسبوع الثامن)."},
    
    {"question": "للحصول على شهادة البكالوريوس في هندسة الذكاء الاصطناعي، ما هو إجمالي الرصيد من الساعات المعتمدة الذي يجب أن أجمعه؟", 
     "ground_truth": "يجب جمع 160 ساعة معتمدة للحصول على درجة البكالوريوس في برامج الساعات المعتمدة."},

    {"question": "أبحث عن الكود الخاص بالمقرر الذي يعلمنا كيفية التخطيط للمشاريع الهندسية وإدارة الموارد البشرية والجودة.", 
     "ground_truth": "كود مقرر إدارة المشروعات هو (حط الكود من اللائحة عندك، غالباً GEN xxx)."},

    {"question": "أريد تسجيل مادة التعلم العميق (Deep Learning)، ما هي المادة التي يجب أن أجتازها أولاً وتتعلق بالشبكات العصبية؟", 
     "ground_truth": "يجب اجتياز مادة الشبكات العصبية أو التعلم الآلي كمتطلب سابق لمادة التعلم العميق."},

    {"question": "هل يمكنك استخراج قائمة بالمقررات الإنسانية أو المهارية التي تزن ساعتين معتمدتين فقط؟", 
     "ground_truth": "قائمة المقررات الإنسانية تشمل (اللغة الإنجليزية، حقوق الإنسان، مهارات التواصل) وكل منها يزن ساعتين."},

    {"question": "هل يُسمح لي بتسجيل مشروع التخرج وبدء العمل فيه خلال الفصل الدراسي الصيفي المكثف؟", 
     "ground_truth": "لا يسمح بتسجيل مشروع التخرج في الفصل الصيفي إلا في حالات استثنائية وبشرط إنهاء المتطلبات."},

    {"question": "كم عدد الأسابيع المطلوبة لإتمام التدريب الميداني الخارجي لطلاب المستوى 300؟", 
     "ground_truth": "يتطلب التدريب الميداني مدة لا تقل عن 4 أسابيع عمل فعلية."},

    {"question": "هل يتيح نظام ابن الهيثم تسجيل المواد الدراسية قبل سداد الرسوم الإدارية ورسوم الخدمات؟", 
     "ground_truth": "يجب سداد الرسوم الإدارية أولاً ليتمكن الطالب من فتح باب التسجيل على نظام ابن الهيثم."},

    {"question": "بعد التخرج، هل سيتم قيدي في نقابة المهندسين تحت شعبة الميكانيكا أم الكهرباء؟", 
     "ground_truth": "يتم القيد حسب التخصص، خريجي حاسبات وذكاء اصطناعي غالباً يتبعون شعبة الكهرباء."},

    {"question": "ما هي الخطوة المالية الأولى المطلوبة لتقديم طلب مراجعة لدرجاتي في امتحان نهائي (Grade Appeal)؟", 
     "ground_truth": "الخطوة الأولى هي سداد رسوم طلب التظلم (مراجعة رصد الدرجات) في خزينة الكلية."},

    {"question": "أنا طالب متعثر (GPA أقل من 2)، ما هو سقف الساعات المعتمدة المسموح لي بتسجيله في الترم الواحد؟", 
     "ground_truth": "يُسمح للطالب المتعثر بتسجيل بحد أقصى 12 ساعة معتمدة في الفصل الدراسي الواحد."},

    {"question": "ما هي الشروط الخاصة للالتحاق بقسم الهندسة النووية في اللائحة؟", 
     "ground_truth": "لا يوجد قسم للهندسة النووية في لائحة كلية الهندسة جامعة المنصورة الحالية."},

    {"question": "هل توفر الكلية باصات خاصة لنقل طلاب الساعات المعتمدة بعد المحاضرات المسائية؟", 
     "ground_truth": "الكلية لا توفر خدمة أتوبيسات خاصة، والخدمة غير مذكورة في اللائحة الرسمية."},

    {"question": "ما هو اسم المادة التي تحمل الكود PDE 052؟", 
     "ground_truth": "كود PDE 052 يخص مادة الرسم الهندسي (أو حسب مسماها باللائحة)."},

    {"question": "ما هو الحد الأدنى للمعدل التراكمي المطلوب لكي يُكتب في شهادتي أنني تخرجت بمرتبة الشرف؟", 
     "ground_truth": "يجب ألا يقل المعدل التراكمي عن 3.0 (أو 3.3 حسب اللائحة) مع عدم الرسوب في أي مادة."},

    {"question": "هل يمكنني استلام شهادة التخرج المؤقتة دون إتمام دورة التربية العسكرية؟", 
     "ground_truth": "لا يمكن استلام شهادة التخرج أو إخلاء الطرف دون إتمام دورة التربية العسكرية للطلاب الذكور."},

    {"question": "إذا رسبت في مادة وحصلت على تقدير F، كم عدد النقاط التي تضاف لمعدلي التراكمي عن هذه المادة؟", 
     "ground_truth": "تقدير F يعادل (صفر) من النقاط، وتدخل الساعات في حساب المعدل التراكمي مما يؤدي لانخفاضه."}
]

def fetch_answer(query: str):
    """دالة لجلب الإجابة من السيرفر بتاعك"""
    try:
        # استبدلنا الـ stream بـ request عادي للتبسيط في التقييم
        response = requests.post(API_URL, json={"text": query, "limit": 3}, timeout=60)
        response.raise_for_status()
        
        # لو السيرفر بيرجع Stream (EventSource)، هنحتاج نفلتر الداتا
        full_answer = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8', errors='ignore')
                if decoded_line.startswith("data: "):
                    content = decoded_line[6:]
                    if content == "[DONE]": break
                    full_answer += content
        
        # ⚠️ سياق افتراضي (Context) - يفضل مستقبلاً ييجي من الـ API
        contexts = ["تتطلب لائحة الكلية 160 ساعة معتمدة. فريق أدفاي يضم محمد، عمر، أمين، وصابر. الكلية بها أقسام حاسبات وميكاترونكس وطبي."]
        
        return full_answer, contexts
    except Exception as e:
        print(f"   ❌ Fetch Error: {e}")
        return "", [""]

def run_ragas_evaluation():
    print("🚀 Starting RAGAS Evaluation Suite for Advi...")
    print("="*50)
    
    # 🌟 هنا تعريف الـ data اللي كان ناقص!
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    for test in TEST_CASES:
        print(f"⏳ Processing: {test['question']}")
        answer, contexts = fetch_answer(test['question'])
        
        if answer:
            print(f"   ✅ Answer: {answer[:60]}...")
        
        eval_data["question"].append(test['question'])
        eval_data["answer"].append(answer)
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append(test['ground_truth'])
        
        time.sleep(1)

    # 3. بناء الـ Dataset
    dataset = Dataset.from_dict(eval_data)

    print("\n🧠 Running RAGAS Metrics (Calculating Faithfulness, Relevancy, etc.)...")
    
    # 4. التقييم الفعلي
    result = evaluate(
        dataset,
        metrics=[
            AnswerRelevancy(), 
            Faithfulness(), 
            ContextPrecision(), 
            ContextRecall()
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    print("\n" + "="*50)
    print("📊 FINAL RAGAS REPORT")
    print("="*50)
    
    df = result.to_pandas()
    
    # 🌟 تعديل ذكي: البحث عن أسماء الأعمدة المتاحة فعلياً
    all_cols = df.columns.tolist()
    # بنحاول نلاقي عمود السؤال (ممكن يكون اسمه question أو user_input)
    q_col = 'question' if 'question' in all_cols else 'user_input'
    
    # الأعمدة اللي عايزين نعرضها لو موجودة
    metrics_cols = ['answer_relevancy', 'faithfulness', 'context_precision', 'context_recall']
    available_metrics = [c for c in metrics_cols if c in all_cols]
    
    # طباعة الجدول باللي لقيناه
    print(df[[q_col] + available_metrics])
    
    print("\n🎯 Summary Scores:")
    # التعديل لنسخة Ragas الحديثة
    scores_dict = result.scores
    for metric_name, values in scores_dict.items():
        # بنحسب المتوسط لكل مادة تقييم
        avg_score = sum(values) / len(values) if values else 0
        print(f"🔹 {metric_name:20}: {avg_score:.4f}")

if __name__ == "__main__":
    run_ragas_evaluation()


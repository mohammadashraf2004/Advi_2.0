import os
import requests
import time
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics.collections import AnswerRelevancy, Faithfulness, ContextPrecision, ContextRecall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY missing in .env")
    exit(1)

BASE_URL  = "http://127.0.0.1:8000/api/v1/nlp"
API_URL   = f"{BASE_URL}/index/answer/1"
CACHE_URL = f"{BASE_URL}/cache/clear"

evaluator_llm        = ChatOpenAI(model="gpt-4o-mini")
evaluator_embeddings = OpenAIEmbeddings()

TEST_CASES = [
    # ── أكاديميات الفصل والإنذار
    {"question": "كم عدد الفصول الدراسية الرئيسية المتتالية التي يظل فيها المعدل أقل من 2.00 قبل أن يُفصل الطالب؟",
     "ground_truth": "يُفصل الطالب إذا ظل معدله التراكمي أقل من 2.00 لمدة 6 فصول دراسية رئيسية متتابعة."},
    {"question": "متى يُنذر الطالب أكاديمياً لأول مرة بسبب المعدل؟",
     "ground_truth": "يُنذر الطالب إذا حصل على معدل تراكمي أقل من 2.00 بنهاية الفصل الدراسي الثاني أو أي فصل لاحق."},
    {"question": "ما الحد الأقصى لساعات تسجيل الطالب المنذر (GPA أقل من 2.00)؟",
     "ground_truth": "يُسمح للطالب المنذر بتسجيل 12 ساعة معتمدة كحد أقصى في الفصل الدراسي."},
    {"question": "هل يُفصل الطالب إذا لم يُكمل متطلبات التخرج في المدة القانونية؟",
     "ground_truth": "نعم، يُفصل الطالب إذا لم يستوفِ شروط التخرج خلال الحد الأقصى للدراسة وهو عشر سنوات."},

    # ── الغياب والمواظبة
    {"question": "ما نسبة الغياب المسموح بها قبل الحرمان من الامتحان النهائي؟",
     "ground_truth": "الحد الأقصى للغياب هو 25% من مجموع ساعات المقرر؛ تجاوزه بدون عذر يُفضي إلى الحرمان من الامتحان ويُحتسب تقدير F."},
    {"question": "عند أي نسبة غياب يصل الطالب الإنذار الأول؟",
     "ground_truth": "يُنذر الطالب للمرة الأولى عند بلوغ نسبة غيابه 10%."},
    {"question": "عند أي نسبة غياب يصل الطالب الإنذار الثاني؟",
     "ground_truth": "يُنذر الطالب للمرة الثانية عند بلوغ نسبة غيابه 20%."},
    {"question": "ما الشرط الطبي للاعتداد بالعذر المرضي عند الانقطاع عن الدراسة؟",
     "ground_truth": "يجب أن تكون الشهادة المرضية صادرة من مستشفى حكومي ومعتمدة من الإدارة الطبية بالجامعة."},

    # ── التسجيل والعبء الدراسي
    {"question": "ما الحد الأقصى لساعات التسجيل للطالب الذي GPA من 2.00 إلى أقل من 3.00؟",
     "ground_truth": "يُسمح له بتسجيل 18 ساعة معتمدة كحد أقصى في الفصل الدراسي."},
    {"question": "ما الحد الأقصى لساعات التسجيل للطالب الذي GPA 3.00 فأكثر؟",
     "ground_truth": "يُسمح له بتسجيل 21 ساعة معتمدة كحد أقصى في الفصل الدراسي."},
    {"question": "ما الحد الأدنى لساعات التسجيل في فصلي الخريف والربيع؟",
     "ground_truth": "الحد الأدنى 12 ساعة معتمدة، ويُستثنى من ذلك حالات التخرج أو التعثر الأكاديمي بموافقة المجلس."},
    {"question": "ما الحد الأقصى لعدد المقررات في الفصل الصيفي؟",
     "ground_truth": "يُسمح بتسجيل مقررين كحد أقصى في الفصل الصيفي، و3 مقررات فقط لحالات التخرج."},
    {"question": "هل يُسمح بتسجيل مشروع التخرج في الفصل الصيفي؟",
     "ground_truth": "لا، يُمنع تسجيل مشاريع التخرج في الفصل الصيفي."},

    # ── الإضافة والحذف والانسحاب
    {"question": "ما الموعد النهائي للإضافة والحذف في الفصول الرئيسية؟",
     "ground_truth": "نهاية الأسبوع الرابع من الفصل الدراسي الرئيسي."},
    {"question": "ما الموعد النهائي للانسحاب الرسمي بتقدير W في الفصول الرئيسية؟",
     "ground_truth": "نهاية الأسبوع العاشر من الفصل الدراسي الرئيسي."},
    {"question": "ما الموعد النهائي للانسحاب بتقدير W في الفصل الصيفي؟",
     "ground_truth": "نهاية الأسبوع الثالث من الفصل الصيفي."},
    {"question": "ما الرمز الذي يظهر في بيان الدرجات عند الانسحاب الرسمي من مقرر؟",
     "ground_truth": "يظهر الرمز (W) دلالةً على الانسحاب الرسمي، ولا يُحتسب في المعدل التراكمي."},

    # ── التقديرات والنقاط
    {"question": "كم نقطة يُعطى لتقدير A+ (97% فأكثر)؟",
     "ground_truth": "تقدير A+ يُعادل 4.00 نقطة."},
    {"question": "ما النسبة المئوية المطلوبة للحصول على تقدير B- ؟",
     "ground_truth": "تقدير B- يُعطى للنسبة من 76% إلى أقل من 80% ويُعادل 2.70 نقطة."},
    {"question": "ما هو تقدير الطالب الذي حصل على 62% في المادة؟",
     "ground_truth": "نسبة 62% تقع ضمن نطاق 60% إلى أقل من 64% وتُعادل تقدير D بنقطة 1.00."},
    {"question": "ما عدد النقاط التي يُعطيها تقدير الرسوب F؟",
     "ground_truth": "تقدير F يُعادل صفر نقطة وتُحتسب ساعاته في قاسم المعدل مما يُخفض المعدل التراكمي."},
    {"question": "ما الشرط الخاص بدرجة الامتحان النهائي للنجاح في أي مقرر؟",
     "ground_truth": "يُشترط الحصول على 40% على الأقل من درجة الامتحان التحريري النهائي كشرط أساسي للنجاح."},
    {"question": "ما النسبة الإجمالية المطلوبة للنجاح في مقرر؟",
     "ground_truth": "يُشترط الحصول على 60% من المجموع الكلي للمقرر للنجاح."},

    # ── مرتبة الشرف والتفوق
    {"question": "ما شروط منح مرتبة الشرف عند التخرج؟",
     "ground_truth": "تُمنح مرتبة الشرف للطالب الحاصل على معدل تراكمي 3.30 فأكثر عند التخرج بشرط عدم الرسوب في أي مقرر طوال فترة الدراسة."},
    {"question": "ما شروط منح شهادة التفوق الفصلية؟",
     "ground_truth": "تُمنح شهادة التفوق للطلاب الحاصلين على معدل 3.60 فأكثر في الفصول السابقة بشرط عدم الرسوب في أي مقرر."},

    # ── نظام التحسين
    {"question": "ما الحد الأقصى لعدد المقررات التي يُسمح للطالب بتحسينها لرفع معدله؟",
     "ground_truth": "يُسمح بتحسين 5 مقررات كحد أقصى خلال فترة الدراسة كاملة."},
    {"question": "أي التقديرين يُعتمد في السجل الأكاديمي عند تحسين مقرر؟",
     "ground_truth": "يُعتمد التقدير الأحدث (الأخير) في السجل الأكاديمي."},
    {"question": "هل يُسمح للطالب بتحسين مقرر رسب فيه بتقدير F؟",
     "ground_truth": "لا، لا يجوز للطالب تحسين مقرر سبق وأن رسب فيه بتقدير F."},
    {"question": "هل يمكن للطالب الانسحاب من مقرر التحسين في أي وقت خلال الفصل الرئيسي؟",
     "ground_truth": "لا يجوز الانسحاب من مقرر التحسين بعد الأسبوع الرابع، لأن البدء في التحسين يترتب عليه محو التقدير الأول."},

    # ── مشروع التخرج والتدريب
    {"question": "كم عدد المشاريع الطلابية الملزم بها طالب هندسة الذكاء الاصطناعي خلال دراسته؟",
     "ground_truth": "يلتزم الطالب بإعداد 2 إلى 3 مشاريع مرتبطة بالصناعة وخدمة المجتمع خلال العامين الدراسيين الأخيرين."},
    {"question": "ما مدة التدريب الميداني لطلاب المستوى 200 (التدريب العملي)؟",
     "ground_truth": "التدريب العملي للمستوى 200 يُجرى داخل الكلية ويستمر أسبوعين بحد أدنى 60 ساعة."},
    {"question": "ما مدة التدريب الميداني لطلاب المستوى 300 و400؟",
     "ground_truth": "التدريب الميداني للمستوى 300 و400 يُجرى خارج الكلية ويستمر 4 أسابيع بحد أدنى 120 ساعة."},
    {"question": "هل يدخل التدريب الميداني في حساب المعدل التراكمي؟",
     "ground_truth": "لا، يُقيَّم التدريب بنظام (ناجح / غير ناجح) ولا تُضاف الدرجات للمعدل التراكمي."},
    {"question": "هل يُقبل التدريب الميداني الأونلاين بالكامل؟",
     "ground_truth": "لا، التدريب الأونلاين بالكامل غير مقبول؛ يجب تقديم طلب رسمي يتضمن مكان التدريب ومدته ومحتواه."},

    # ── ساعات التخرج والتوزيع
    {"question": "ما إجمالي الساعات المعتمدة المطلوبة للحصول على البكالوريوس في هندسة الذكاء الاصطناعي؟",
     "ground_truth": "يجب اجتياز 160 ساعة معتمدة كحد أدنى."},
    {"question": "ما إجمالي ساعات متطلبات الجامعة في برنامج الذكاء الاصطناعي؟",
     "ground_truth": "13 ساعة معتمدة موزعة على 7 مقررات (8% من الإجمالي)."},
    {"question": "ما إجمالي ساعات متطلبات الكلية في برنامج الذكاء الاصطناعي؟",
     "ground_truth": "45 ساعة معتمدة موزعة على 16 مقرراً إلزامياً."},
    {"question": "كم عدد المقررات الاختيارية المطلوبة ضمن متطلبات التخصص الدقيق؟",
     "ground_truth": "5 مقررات اختيارية بإجمالي 15 ساعة معتمدة."},

    # ── المقررات والأكواد
    {"question": "ما الذي يرمز إليه الحرف الأول من الأرقام الثلاثة في كود المقرر؟",
     "ground_truth": "يدل على المستوى الدراسي (مثلاً: 1 للمستوى 100، و4 للمستوى 400)."},
    {"question": "ما الدلالة الكاملة لكود CSE في منظومة تكويد المقررات؟",
     "ground_truth": "CSE يرمز إلى قسم هندسة الحاسبات ونظم التحكم."},
    {"question": "ما الدلالة الكاملة لكود UNR في منظومة تكويد المقررات؟",
     "ground_truth": "UNR يرمز إلى متطلبات الجامعة (University Requirements)."},
    {"question": "ما اسم المقرر الذي يحمل الكود BAS 011؟",
     "ground_truth": "BAS 011 هو مقرر رياضيات (1) بواقع 3 ساعات معتمدة."},
    {"question": "ما اسم مقرر التخرج الأول الذي يتضمنه هيكل برنامج الذكاء الاصطناعي وما كوده؟",
     "ground_truth": "مشاريع التخرج تحمل الأكواد ARI 381 وARI 481 وARI 482 بإجمالي 9 ساعات معتمدة."},

    # ── التظلمات والرسوم
    {"question": "كم المهلة الزمنية المتاحة للطالب لتقديم طلب مراجعة درجاته (التظلم) بعد إعلان النتيجة؟",
     "ground_truth": "يحق للطالب تقديم طلب مراجعة خلال أسبوع واحد من إعلان النتيجة بعد سداد الرسوم المقررة."},
    {"question": "ما تكلفة الساعة المعتمدة في برنامج الذكاء الاصطناعي للعام 2025/2026؟",
     "ground_truth": "تكلفة الساعة المعتمدة 1330 جنيهاً مصرياً."},
    {"question": "ما الرسوم الإدارية الثابتة المضافة لكل فصل دراسي في برنامج الذكاء الاصطناعي؟",
     "ground_truth": "الرسوم الإدارية الثابتة 2089 جنيهاً مصرياً لكل فصل دراسي."},

    # ── اللوجستيات والخدمات
    {"question": "ما الوضع النقابي لخريج برنامج هندسة الذكاء الاصطناعي؟",
     "ground_truth": "يُقيَّد الخريج في نقابة المهندسين المصرية تحت شعبة كهرباء، والمسمى الرسمي: مهندس ذكاء اصطناعي (شعبة كهرباء)."},
    {"question": "هل تتوفر أتوبيسات خاصة لنقل طلاب البرامج النوعية؟",
     "ground_truth": "لا توجد أتوبيسات خاصة، لكن الكلية تحرص على إنهاء اليوم الدراسي مبكراً وتوفر الجامعة سكناً داخل الحرم وخارجه."},
    {"question": "هل تقبل الكلية معادلة كورسات منصات مثل Coursera أو Udacity بدلاً من المقررات الرسمية؟",
     "ground_truth": "لا، لا تقبل الكلية معادلة كورسات المنصات العالمية (MOOCs) كبديل للمقررات الرسمية."},
    {"question": "هل اجتياز دورة التربية العسكرية شرط أساسي لاستلام شهادة التخرج؟",
     "ground_truth": "نعم، التربية العسكرية شرط إلزامي لاستلام شهادة التخرج وإخلاء الطرف للطلاب الذكور."},

    # ── المقررات غير المكتملة (I)
    {"question": "ما المهلة المتاحة لتقديم عذر الغياب عن الامتحان النهائي للحصول على تقدير I (غير مكتمل)؟",
     "ground_truth": "يجب تقديم العذر للمجلس الأكاديمي ومجلس الكلية خلال يومين كحد أقصى من تاريخ الامتحان."},
    {"question": "ما الشرط الأكاديمي لاستحقاق تقدير I (غير مكتمل)؟",
     "ground_truth": "يجب أن يكون الطالب حاصلاً على 60% على الأقل من درجات أعمال السنة وألا يكون قد حُرم من دخول الامتحان."},

    # ── متفرقات
    {"question": "ما نسبة الحضور الدراسي المطلوبة في برنامج الذكاء الاصطناعي؟",
     "ground_truth": "يُشترط حضور 75% على الأقل من الساعات الفعلية للمحاضرات والسكاشن والمعامل."},
    {"question": "ما الحد الأقصى لفترة انقطاع الطالب التي يُسمح بعدها بإعادة التسجيل؟",
     "ground_truth": "يجوز إعادة التسجيل لمن ترك الدراسة لمدة تصل إلى 4 فصول دراسية بحد أقصى، بشرط الحصول على تقديرات عالية سابقاً وموافقة المجلس الأكاديمي."},
]


# ── helpers ───────────────────────────────────────────────────────────────────

def clear_eval_history():
    try:
        import pymongo
        client = pymongo.MongoClient(
            "mongodb://admin:admin@localhost:27007/mini_rag?authSource=admin"
        )
        db = client["Advi_db"]
        
        # ← clear ALL history, not just project_id=1, to be safe
        deleted = db["chat_history"].delete_many({})
        print(f"✅ Cleared {deleted.deleted_count} chat history records")
        client.close()
    except Exception as e:
        print(f"⚠️  Could not clear chat history: {e}")


def clear_server_cache():
    """
    مسح الـ in-memory SemanticCache داخل السيرفر.
    ضروري لأن الكاش مشترك طوال عمر السيرفر — بدونه
    أسئلة متشابهة دلالياً تُعيد نتائج BM25/vector من سؤال سابق.
    """
    try:
        resp = requests.post(CACHE_URL, timeout=10)
        resp.raise_for_status()
        print(f"✅ Server in-memory cache cleared: {resp.json()}")
    except Exception as e:
        print(f"⚠️  Could not clear server cache: {e}")


def fetch_answer(query: str):
    session = requests.Session()
    try:
        response = session.post(
            API_URL,
            json={
                "text":     query,
                "limit":    3,
                "no_cache": True,   # skip Qdrant answer-cache read/write
                "raw_mode": True,   # skip MongoDB history read/write
            },
            timeout=90,
            stream=True,
        )
        response.raise_for_status()

        full_answer = ""
        lines_read  = 0

        for line in response.iter_lines():
            if line:
                decoded = line.decode("utf-8", errors="ignore")
                lines_read += 1
                if decoded.startswith("data: "):
                    content = decoded[6:]
                    if content == "[DONE]":
                        print(f"   [DEBUG] [DONE] lines_read={lines_read}")
                        break
                    full_answer += content

        if not full_answer:
            print(f"   [DEBUG] empty answer — lines_read={lines_read}")
        else:
            print(f"   [DEBUG] lines_read={lines_read} | answer_len={len(full_answer)}")

        contexts = [
            "لائحة كلية الهندسة جامعة المنصورة – نظام الساعات المعتمدة. "
            "الحد الأقصى للغياب 25%، الطالب المنذر يُسجل 12 ساعة كحد أقصى، "
            "الفصل بعد 6 فصول رئيسية متتابعة بمعدل أقل من 2.00. "
            "برنامج الذكاء الاصطناعي 160 ساعة معتمدة، مرتبة الشرف بمعدل 3.30 فأكثر."
        ]
        return full_answer, contexts

    except Exception as e:
        print(f"   ❌ Fetch Error: {e}")
        return "", [""]
    finally:
        session.close()


# ── main ──────────────────────────────────────────────────────────────────────

def run_ragas_evaluation():
    print("🚀 Starting RAGAS Evaluation – Advi RAG System")
    print(f"   Total test cases: {len(TEST_CASES)}")
    print("=" * 60)

    clear_eval_history()   # step 1: wipe MongoDB history
    clear_server_cache()   # step 2: wipe in-memory SemanticCache

    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n{'-'*60}")
        print(f"[{i:02d}/{len(TEST_CASES)}] ❓ {test['question']}")
        print(f"            ✔️  {test['ground_truth']}")

        answer, contexts = fetch_answer(test["question"])
        print(f"            🤖 {answer}" if answer else "            ⚠️  (no answer)")

        eval_data["question"].append(test["question"])
        eval_data["answer"].append(answer)
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append(test["ground_truth"])
        time.sleep(2)

    dataset = Dataset.from_dict(eval_data)
    print("\n🧠 Running RAGAS Metrics …")

    result = evaluate(
        dataset,
        metrics=[AnswerRelevancy(), Faithfulness(), ContextPrecision(), ContextRecall()],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    print("\n" + "=" * 60)
    print("📊 FINAL RAGAS REPORT")
    print("=" * 60)

    df           = result.to_pandas()
    all_cols     = df.columns.tolist()
    q_col        = "question" if "question" in all_cols else "user_input"
    ans_col      = "answer"   if "answer"   in all_cols else "response"
    metrics_cols = ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]
    available    = [c for c in metrics_cols if c in all_cols]
    display_cols = [q_col] + ([ans_col] if ans_col in all_cols else []) + available

    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 220)
    print(df[display_cols].to_string(index=False))

    print("\n🎯 Summary – Average Scores:")
    scores_raw = result.scores
    if isinstance(scores_raw, list):
        for metric in scores_raw[0].keys():
            vals = [r[metric] for r in scores_raw if r.get(metric) is not None]
            print(f"   🔹 {metric:<25}: {sum(vals)/len(vals):.4f}" if vals else f"   🔹 {metric:<25}: N/A")
    else:
        for name, vals in scores_raw.items():
            print(f"   🔹 {name:<25}: {sum(vals)/len(vals):.4f}" if vals else f"   🔹 {name:<25}: N/A")

    out = "ragas_evaluation_results.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n💾 Saved to: {out}")


if __name__ == "__main__":
    run_ragas_evaluation()
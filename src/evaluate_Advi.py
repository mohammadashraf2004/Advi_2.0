# -*- coding: utf-8 -*-
import requests
import time
import json
from openai import OpenAI
import os
import sys

# 🌟 إجبار البايثون على استخدام UTF-8 لتجنب مشاكل الـ Linux/WSL
if sys.platform.startswith('linux') or sys.platform == 'darwin':
    os.environ["PYTHONIOENCODING"] = "utf-8"

# ==========================================
# 📊 ADVI RAG EVALUATION SCRIPT
# ==========================================

# 1. إعدادات السيرفر والـ API
API_URL = "http://127.0.0.1:8000/api/v1/nlp/index/answer/1"

# ⚠️ تأكد من وضع مفتاح OpenAI الحقيقي الخاص بك هنا
import os
import sys
from dotenv import load_dotenv # 🌟 استدعاء المكتبة

# 🌟 قراءة ملف .env تلقائياً
load_dotenv() 

# إجبار البايثون على استخدام UTF-8
if sys.platform.startswith('linux') or sys.platform == 'darwin':
    os.environ["PYTHONIOENCODING"] = "utf-8"

# ... (باقي الكود زي ما هو) ...

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# 2. حزمة الاختبار (Test Suite)
TEST_CASES = [
    {
        "query": "إيه هي متطلبات التخرج من كلية الهندسة؟",
        "expected_keywords": ["160", "ساعة معتمدة", "اجتياز"],
        "intent": "ACADEMIC"
    },
    {
        "query": "مين هما التيم اللي عمل مشروع أدفاي؟",
        "expected_keywords": ["عمر أشرف", "أمين محمد", "صابر محمود", "محمد أشرف"],
        "intent": "ACADEMIC"
    },
    {
        "query": "إيه هي أقسام الكلية المتاحة؟",
        "expected_keywords": ["حاسبات", "ميكاترونكس", "إلكترونيات", "طبي"],
        "intent": "ACADEMIC"
    },
    {
        "query": "إزاي أكتب سيرة ذاتية (CV) قوية للشركات؟",
        "expected_keywords": ["سيرة ذاتية", "خبرات", "مشاريع"],
        "intent": "JOB"
    },
    {
        "query": "عاوز كورس كويس أتعلم منه Machine Learning من الصفر",
        "expected_keywords": ["كورس", "Coursera", "Udemy", "تعلم الآلة"],
        "intent": "COURSE"
    }
]

def fetch_streaming_answer(query: str):
    start_time = time.time()
    try:
        response = requests.post(
            API_URL, 
            json={"text": query, "limit": 5},
            stream=True,
            timeout=20
        )
        response.raise_for_status()
        
        full_answer = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8', errors='ignore')
                if decoded_line.startswith("data: "):
                    data = decoded_line[6:]
                    if data == "[DONE]":
                        break
                    full_answer += data
                    
        latency = time.time() - start_time
        return full_answer, latency
    except Exception as e:
        print(f"\n   ❌ Fetch Error: {e}")
        return "", 0.0

def evaluate_answer_with_llm(query: str, generated_answer: str, expected_keywords: list):
    # تحويل المصفوفة لنص مباشر لمنع مشاكل الـ Encoding
    expected_str = ", ".join(expected_keywords)
    
    judge_prompt = f"""أنت مقيّم جودة صارم لنظام أسئلة وأجوبة (RAG) جامعي.
سؤال المستخدم: "{query}"
الإجابة التي ولدها النظام: "{generated_answer}"
الكلمات/المفاهيم المفتاحية التي يجب أن تتضمنها الإجابة: {expected_str}

هل الإجابة دقيقة، مفيدة، وتغطي المفاهيم المطلوبة؟
رد بصيغة JSON فقط تحتوي على:
"score": درجة من 10 (رقم صحيح)
"reason": سبب التقييم باختصار شديد

JSON:"""

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": judge_prompt}],
            response_format={ "type": "json_object" },
            temperature=0.0
        )
        content = res.choices[0].message.content.strip()
        
        # حماية إضافية لو الموديل أضاف علامات Markdown للـ JSON
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
            
        evaluation = json.loads(content)
        return evaluation.get("score", 0), evaluation.get("reason", "No reason provided")
    except Exception as e:
        return 0, str(e)

def run_evaluation():
    print("🚀 Starting Advi Evaluation Suite...\n")
    print("="*50)
    
    total_score = 0
    total_latency = 0
    results = []

    for i, test in enumerate(TEST_CASES):
        print(f"⏳ Testing [{i+1}/{len(TEST_CASES)}]: {test['query']}")
        
        answer, latency = fetch_streaming_answer(test['query'])
        
        if not answer:
            print("   ❌ Failed to get answer.\n")
            continue
            
        score, reason = evaluate_answer_with_llm(test['query'], answer, test['expected_keywords'])
        
        total_score += score
        total_latency += latency
        
        results.append({
            "query": test['query'],
            "score": score,
            "latency": latency,
            "reason": reason
        })
        
        print(f"   ✅ Score: {score}/10 | Latency: {latency:.2f}s")
        print(f"   📝 Reason: {reason}\n")

    if results:
        avg_score = total_score / len(results)
        avg_latency = total_latency / len(results)
        
        print("="*50)
        print("📊 EVALUATION REPORT")
        print("="*50)
        print(f"🎯 Average Accuracy Score: {avg_score:.1f} / 10")
        print(f"⚡ Average Latency: {avg_latency:.2f} seconds")
        print(f"✅ Total Tests Run: {len(results)}")
        
        if avg_score >= 8.5:
            print("\n🌟 SYSTEM STATUS: EXCELLENT (Production Ready)")
        elif avg_score >= 7.0:
            print("\n👍 SYSTEM STATUS: GOOD (Needs minor tuning)")
        else:
            print("\n⚠️ SYSTEM STATUS: POOR (Needs major improvements)")
        print("="*50)

if __name__ == "__main__":
    run_evaluation()
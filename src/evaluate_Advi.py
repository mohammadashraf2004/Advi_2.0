# -*- coding: utf-8 -*-
import requests
import time
import json
from openai import OpenAI
import os
import sys
from dotenv import load_dotenv

# 🌟 إجبار البايثون على استخدام UTF-8 لتجنب مشاكل الـ Linux/WSL
if sys.platform.startswith('linux') or sys.platform == 'darwin':
    os.environ["PYTHONIOENCODING"] = "utf-8"

# قراءة ملف .env تلقائياً
load_dotenv() 

# ==========================================
# 📊 ADVI RAG EVALUATION SCRIPT WITH BILLING
# ==========================================

# 1. إعدادات السيرفر والـ API
API_URL = "http://127.0.0.1:8000/api/v1/nlp/index/answer/1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("⚠️ تحذير: مفتاح OPENAI_API_KEY غير موجود في ملف .env")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

# تسعيرة gpt-4o-mini (لكل مليون توكن) - حسب أسعار OpenAI
GPT4O_MINI_INPUT_PRICE_PER_1M = 0.150
GPT4O_MINI_OUTPUT_PRICE_PER_1M = 0.600

# 2. حزمة الاختبار (Test Suite) - 15 سؤال إجمالاً
TEST_CASES = [
    # --- الأسئلة الأكاديمية العامة والقديمة ---
    {"query": "إيه هي متطلبات التخرج من كلية الهندسة؟", "expected_keywords": ["ساعة معتمدة", "اجتياز"], "intent": "ACADEMIC"},
    {"query": "مين هما التيم اللي عمل مشروع أدفاي؟", "expected_keywords": ["عمر أشرف", "أمين محمد", "صابر محمود"], "intent": "ACADEMIC"},
    {"query": "إيه هي أقسام الكلية المتاحة؟", "expected_keywords": ["حاسبات", "ميكاترونكس", "إلكترونيات", "طبي"], "intent": "ACADEMIC"},
    {"query": "إزاي أكتب سيرة ذاتية (CV) قوية للشركات؟", "expected_keywords": ["سيرة ذاتية", "خبرات", "مشاريع"], "intent": "JOB"},
    {"query": "عاوز كورس كويس أتعلم منه Machine Learning من الصفر", "expected_keywords": ["كورس", "Coursera", "Udemy"], "intent": "COURSE"},
    
    # --- الـ 10 أسئلة الجديدة (مخصصة للذكاء الاصطناعي، الحسابات، والتطوير) ---
    {"query": "لو هسجل 15 ساعة في قسم الذكاء الاصطناعي، هدفع كام بالظبط؟", "expected_keywords": ["22039", "2089", "1330", "جنيهاً"], "intent": "ACADEMIC"},
    {"query": "هل اللائحة بتسمحلي أتخرج من قسم الذكاء الاصطناعي في 4 سنين بس بدل 5؟", "expected_keywords": ["4 سنوات", "تخرج", "لائحة", "مستويات"], "intent": "ACADEMIC"},
    {"query": "إيه هي المواد الإجبارية في قسم الذكاء الاصطناعي؟", "expected_keywords": ["إجبارية", "مقررات", "الذكاء الاصطناعي"], "intent": "ACADEMIC"},
    {"query": "مين دكتور مادة الـ Machine Learning؟", "expected_keywords": ["غير مذكور", "دكتور", "اللائحة"], "intent": "ACADEMIC"},
    {"query": "عايز اشتغل AI Engineer، إيه المهارات المطلوبة في سوق العمل المصري؟", "expected_keywords": ["Python", "Machine Learning", "NLP", "مهارات"], "intent": "JOB"},
    {"query": "إيه هي فرص العمل المتاحة لخريجي قسم الذكاء الاصطناعي؟", "expected_keywords": ["مهندس", "Data Scientist", "شركات"], "intent": "JOB"},
    {"query": "عايز نصيحة عشان أظبط بروفايل لينكد إن بتاعي كمهندس AI", "expected_keywords": ["LinkedIn", "مشاريع", "خبرات", "تواصل"], "intent": "JOB"},
    {"query": "عايز كورس لتعليم الـ Natural Language Processing و Large Language Models", "expected_keywords": ["كورس", "Coursera", "NLP", "LLM"], "intent": "COURSE"},
    {"query": "عايز كورسات لتعلم الـ Computer Vision وتحليل الصور الطبية", "expected_keywords": ["Computer Vision", "صور طبية", "Udemy", "كورس"], "intent": "COURSE"},
    {"query": "إيه هو التحديث الجديد في مشروع ADVI 2.0؟", "expected_keywords": ["Voice", "وكلاء", "Scraping", "RAG"], "intent": "ACADEMIC"}
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
        
        # حساب التوكنز والتكلفة
        prompt_tokens = res.usage.prompt_tokens
        completion_tokens = res.usage.completion_tokens
        
        cost = (prompt_tokens * GPT4O_MINI_INPUT_PRICE_PER_1M / 1_000_000) + \
               (completion_tokens * GPT4O_MINI_OUTPUT_PRICE_PER_1M / 1_000_000)

        content = res.choices[0].message.content.strip()
        
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
            
        evaluation = json.loads(content)
        return evaluation.get("score", 0), evaluation.get("reason", "No reason provided"), prompt_tokens, completion_tokens, cost
    except Exception as e:
        return 0, str(e), 0, 0, 0.0

def run_evaluation():
    print("🚀 Starting Advi Evaluation Suite...\n")
    print("="*60)
    
    total_score = 0
    total_latency = 0
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    results = []

    for i, test in enumerate(TEST_CASES):
        print(f"⏳ Testing [{i+1}/{len(TEST_CASES)}]: {test['query']}")
        
        answer, latency = fetch_streaming_answer(test['query'])
        
        if not answer:
            print("   ❌ Failed to get answer.\n")
            continue

        # ==========================================
        # 🟢 الجزء الجديد: طباعة الإجابة في التيرمينال
        # ==========================================
        print("\n   🤖 System Answer:")
        print(f"   {'-'*40}")
        for line in answer.split('\n'):
            print(f"   {line}")
        print(f"   {'-'*40}\n")
        # ==========================================
            
        score, reason, p_tokens, c_tokens, cost = evaluate_answer_with_llm(test['query'], answer, test['expected_keywords'])
        
        total_score += score
        total_latency += latency
        total_cost += cost
        total_prompt_tokens += p_tokens
        total_completion_tokens += c_tokens
        
        results.append({
            "query": test['query'],
            "score": score,
            "latency": latency,
            "reason": reason,
            "cost": cost
        })
        
        print(f"   ✅ Score: {score}/10 | Latency: {latency:.2f}s")
        print(f"   💸 Tokens: {p_tokens} (in) + {c_tokens} (out) | Cost: ${cost:.6f}")
        print(f"   📝 Reason: {reason}\n")

    if results:
        avg_score = total_score / len(results)
        avg_latency = total_latency / len(results)
        
        print("="*60)
        print("📊 EVALUATION REPORT & BILLING")
        print("="*60)
        print(f"🎯 Average Accuracy Score: {avg_score:.1f} / 10")
        print(f"⚡ Average Latency: {avg_latency:.2f} seconds")
        print(f"✅ Total Tests Run: {len(results)}")
        print("-" * 60)
        print("💰 BILLING SUMMARY (OpenAI Evaluator - gpt-4o-mini)")
        print(f"   Input Tokens:  {total_prompt_tokens}")
        print(f"   Output Tokens: {total_completion_tokens}")
        print(f"   Total Cost:    ${total_cost:.6f}")
        print("-" * 60)
        
        if avg_score >= 8.5:
            print("🌟 SYSTEM STATUS: EXCELLENT (Production Ready)")
        elif avg_score >= 7.0:
            print("👍 SYSTEM STATUS: GOOD (Needs minor tuning)")
        else:
            print("⚠️ SYSTEM STATUS: POOR (Needs major improvements)")
        print("="*60)

if __name__ == "__main__":
    run_evaluation()
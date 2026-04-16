import json
import os
import re
import uuid
from datetime import datetime
from .BaseAgent import BaseAgent
from models.db_schemas import Project
from qdrant_client.http.models import PointStruct

# courses_db.json is in the same directory as this agent file
_COURSES_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'courses_db.json')


class VectorDBAgent(BaseAgent):
    def __init__(self, vectordb_client, generation_client, mongo_client,
                 template_parser, embedding_client, reranker_client=None):
        super().__init__(vectordb_client, generation_client, mongo_client,
                         template_parser, embedding_client)
        self.reranker_client = reranker_client
        self.cache_threshold = 0.92
        self._qdrant         = getattr(vectordb_client, 'client', vectordb_client)

        from controllers.NLPController import NLPController
        self.nlp_controller = NLPController(
            vectordb_client=vectordb_client,
            generation_client=generation_client,
            embedding_client=embedding_client,
            template_parser=template_parser,
            mongo_client=mongo_client,
            reranker_client=reranker_client,
        )

        # Load courses once at init — no re-read on every request
        self._courses_dict = {}
        candidates = [
            _COURSES_DB_PATH,
            os.path.join(os.getcwd(), 'courses_db.json'),
            os.path.join(os.getcwd(), 'src', 'agents', 'courses_db.json'),
            'courses_db.json',
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        self._courses_dict = json.load(f).get('courses', {})
                    print(f"[DEBUG] Loaded {len(self._courses_dict)} courses from: {path}")
                    break
                except Exception as e:
                    print(f"[DEBUG] Failed to load courses from {path}: {e}")
        else:
            print(f"[WARN] courses_db.json not found. Searched: {candidates}")

        print("[DEBUG] VectorDBAgent initialized successfully.")

    # ── Text cleanup ──────────────────────────────────────────────────────
    def clean_output_text(self, text: str) -> str:
        if not text:
            return ""
        # Frontend (marked.js) handles markdown — only strip technical leaks
        text = re.sub(r'(العنوان:|Title:|المحتوى:|تفاصيل السطر:)', '', text)
        text = re.sub(r' +', ' ', text)
        return text

    # ── JSON course lookup ────────────────────────────────────────────────
    def _find_course(self, query: str):
        if not self._courses_dict:
            return None

        norm_query       = self.nlp_controller.normalize_arabic(query.lower())
        norm_query_clean = re.sub(r'[^\w\s]', ' ', norm_query).strip()

        print(f"[DEBUG] Interceptor norm_query: '{norm_query_clean}'")

        for code, info in self._courses_dict.items():
            norm_name = self.nlp_controller.normalize_arabic(
                info.get('name', '').lower()
            ).strip()
            norm_code = re.sub(r'\s+', '', code.lower())

            name_match = bool(norm_name) and norm_name in norm_query_clean
            code_match = norm_code in norm_query_clean.replace(' ', '')

            if name_match or code_match:
                result         = dict(info)
                result['code'] = code
                print(f"[DEBUG] JSON matched: {info.get('name','').strip()} ({code})")
                return result

        print(f"[DEBUG] No JSON match for: '{norm_query_clean}'")
        return None

    def _build_course_context(self, course_data: dict) -> str:
        code    = course_data['code']
        name    = course_data.get('name', '').strip()
        credits = course_data.get('credits', '?')
        prereqs = course_data.get('prerequisites', [])

        if prereqs:
            parts = []
            for p_code in prereqs:
                p_name = self._courses_dict.get(p_code, {}).get('name', p_code).strip()
                parts.append(f"{p_name} ({p_code})")
            prereq_string = "، ".join(parts)
        else:
            prereq_string = "لا يوجد متطلب سابق"

        print(f"[DEBUG] JSON hit: {name} ({code}) — prereqs: {prereq_string}")

        return (
            f"--- بيانات رسمية من ملف المقررات ---\n"
            f"كود المقرر: {code}\n"
            f"اسم المقرر: {name}\n"
            f"الساعات المعتمدة: {credits}\n"
            f"المتطلب السابق: {prereq_string}\n"
        )

    # ── Cache save helper ─────────────────────────────────────────────────
    def _save_to_qdrant_cache(self, qdrant_client, query_vector,
                               optimized_query, full_answer):
        try:
            qdrant_client.upsert(
                collection_name="semantic_cache",
                points=[PointStruct(
                    id=str(uuid.uuid4()),
                    vector=query_vector,
                    payload={
                        "query":     optimized_query,
                        "answer":    full_answer,
                        "timestamp": datetime.now().isoformat()
                    }
                )]
            )
            print("[DEBUG] ✅ Saved to cache.")
        except Exception as e:
            print(f"[DEBUG] Cache save failed: {e}")

    # ── Critic evaluator ──────────────────────────────────────────────────
    async def evaluate_answer(self, query: str, context: str,
                               draft_answer: str) -> bool:
        prompt = f"""You are a strict RAG evaluator.
Task: Determine if the 'Draft Answer' is grounded in the 'Context'.
Rules:
- Answer is correct and supported by context → "PASS"
- Answer says information is not found AND context truly doesn't contain it → "PASS"
- Answer contains hallucinated facts not present in context → "FAIL"
- Answer says "not found" when information IS clearly in context → "FAIL"
- NO EXPLANATIONS. ONLY ONE WORD: PASS or FAIL.

Query: {query}
Context: {context}
Draft Answer: {draft_answer}
Decision:"""
        try:
            result   = self.generation_client.generate_response(
                prompt=prompt,
                chat_history=[],
                temperature=0.0,
                max_output_tokens=2
            )
            decision = result.strip().upper()
            print(f"[DEBUG] Critic: {decision}")
            return "PASS" in decision
        except Exception as e:
            print(f"[DEBUG] Critic error: {e}")
            return True  # fail open on technical errors

    # ── Main stream ───────────────────────────────────────────────────────
    async def process_stream(self, project: Project, query: str,
                             chat_history: list = None, limit: int = 5,
                             skip_evaluation: bool = False):
        print(f"\n[DEBUG] === Agent Stream: '{query}' ===")

        if chat_history is None:
            chat_history = []

        optimized_query = query

        # ── 1. Qdrant semantic cache ──────────────────────────────────────
        query_vector  = self.embedding_client.embed_text(optimized_query)
        qdrant_client = getattr(self.vectordb_client, 'client', self.vectordb_client)

        try:
            cache_results = qdrant_client.search(
                collection_name="semantic_cache",
                query_vector=query_vector,
                limit=1
            )
            if cache_results and cache_results[0].score >= self.cache_threshold:
                print(f"[DEBUG] Cache HIT! Score: {cache_results[0].score:.4f}")
                yield cache_results[0].payload.get('answer', '')
                return
        except Exception as e:
            print(f"[DEBUG] Cache lookup failed (may not exist yet): {e}")

        # ── 2. Financial interceptor ──────────────────────────────────────
        if re.search(r'(ادفع|مصاريف|تكلفة|حساب|رسوم|سعر|فلوس|أدفع|إدفع)',
                     optimized_query, re.IGNORECASE):
            print("[DEBUG] Financial intent detected.")
            hours_match = re.search(r'(\d+)\s*(ساعة|ساعات|ساعه)', optimized_query)
            if hours_match:
                hours  = int(hours_match.group(1))
                total  = 2089 + (1330 * hours)
                answer = (
                    f"💰 **حساب المصروفات الدراسية:**\n\n"
                    f"لتسجيل **{hours} ساعة** معتمدة، الحسبة كالتالي:\n"
                    f"- **الرسوم الإدارية الثابتة:** 2089 جنيهاً\n"
                    f"- **تكلفة الساعات:** {hours} × 1330 = {1330 * hours} جنيهاً\n"
                    f"- **الإجمالي التقديري:** **{total} جنيهاً مصرياً**"
                )
            else:
                answer = (
                    f"💰 **حساب المصروفات الدراسية:**\n\n"
                    f"المصاريف تُحسب بناءً على عدد الساعات المسجلة:\n"
                    f"- **سعر الساعة:** 1330 جنيهاً\n"
                    f"- **رسوم إدارية ثابتة:** 2089 جنيهاً\n"
                    f"- **المعادلة:** 2089 + (عدد الساعات × 1330)"
                )
            yield answer
            return

        # ── 3. JSON course interceptor ────────────────────────────────────
        course_data  = self._find_course(optimized_query)
        context_text = ""
        final_docs   = []

        # ── 4. Build context ──────────────────────────────────────────────
        if course_data:
            context_text = self._build_course_context(course_data)
        else:
            print("[DEBUG] No JSON match — searching VectorDB...")
            final_docs = await self.nlp_controller.search_vector_db_collection(
                project=project,
                text=optimized_query,
                limit=limit,
                chat_history=chat_history,
            )
            if not final_docs:
                yield "غير مذكور في اللائحة"
                return

            context_text = "\n\n".join([
                f"--- مقتبس {i+1} ---\n{doc.text}"
                for i, doc in enumerate(final_docs[:5])
            ])

            print("\n" + "📚 " * 10 + " RETRIEVED CHUNKS " + "📚 " * 10)
            for i, doc in enumerate(final_docs[:4]):
                print(f"👉 [Chunk {i+1}] | Score: {doc.score:.4f}")
                print(f"📝 {doc.text[:120]}")
                print("-" * 50)

        # ── 5. Build LLM history ──────────────────────────────────────────
        system_prompt = """أنت "الزميل المساعد والمرشد الأكاديمي الذكي" لبرنامج هندسة الذكاء الاصطناعي بجامعة المنصورة.
مهمتك هي تحويل اللوائح الجافة إلى نصائح واضحة، صديقة للمستخدم، ودقيقة بنسبة 100%.

⚠️ القواعد الذهبية:
1. اعتمد فقط على المقتبسات المتاحة. إذا لم تجد المعلومة قل: "هذه المعلومة غير مذكورة في اللائحة الحالية".
2. عندما تجد سطراً بصيغة (X - Y - Z) فاعلم أنه سطر من جدول مفرود.
3. "دواير" = "دوائر"، "Level 100" = "المستوى 100". لا تكن حرفياً.

🔑 قاعدة الجملة الأولى:
ابدأ ردك دائماً بذكر اسم الموضوع أو المقرر صراحةً في الجملة الأولى.
مثال صحيح: "مقرر تعلم الآلة (CSE 251) هو مقرر إجباري بثلاث ساعات معتمدة..."
السبب: تُستخدم لاحقاً لفهم الضمائر في الأسئلة التالية.

✍️ أسلوب الرد:
- ابدأ بالإجابة المباشرة ثم فصّل بالنقاط.
- استخدم جداول Markdown للمقررات والدرجات.
- استخدم الخط العريض للكلمات المفتاحية فقط.
- نبّه فوراً لقاعدة الـ 25% والحرمان إذا سأل عن الغياب.

💰 المصاريف = 2089 + (عدد الساعات × 1330).

يمنع ذكر: Metadata، Chunk، تفاصيل السطر."""

        final_chat_history = [
            self.generation_client.construct_prompt(prompt=system_prompt, role="system")
        ]
        for msg in chat_history:
            final_chat_history.append(
                self.generation_client.construct_prompt(
                    prompt=msg['content'], role=msg['role']
                )
            )

        # ── 6. Final prompt ───────────────────────────────────────────────
        full_prompt = (
            f"📚 السياق المستخرج:\n{context_text}\n\n"
            f"🎯 السؤال الحالي:\n{query}\n\n"
            f"✍️ الإجابة المباشرة:"
        )

        stream_generator = self.generation_client.generate_stream(
            prompt=full_prompt,
            chat_history=final_chat_history,
            temperature=0.0,
            max_output_tokens=2048
        )

        # ── 7. Stream & collect ───────────────────────────────────────────
        in_thinking_block     = False
        full_answer_for_cache = ""
        buffer                = ""

        async for chunk in stream_generator:
            if "<thinking>" in chunk:
                in_thinking_block = True
                chunk = chunk.replace("<thinking>", "")

            if in_thinking_block:
                if "</thinking>" in chunk:
                    after             = chunk.split("</thinking>", 1)[1]
                    in_thinking_block = False
                    clean             = self.clean_output_text(after)
                    if clean:
                        full_answer_for_cache += clean
                        yield clean
                else:
                    buffer += chunk.replace("</thinking>", "")
            else:
                clean_chunk = self.clean_output_text(chunk)
                if clean_chunk:
                    full_answer_for_cache += clean_chunk
                    yield clean_chunk

        # Flush buffer if </thinking> was never emitted
        if buffer.strip():
            clean = self.clean_output_text(buffer)
            if clean:
                full_answer_for_cache += clean
                yield clean

        # ── 8. Evaluate and cache ─────────────────────────────────────────
        # JSON-intercepted answers are authoritative — skip
        # Voice mode skips evaluation for speed
        if full_answer_for_cache.strip() and not course_data and not skip_evaluation:
            max_score = max((doc.score for doc in final_docs), default=0.0)

            if max_score < 0.05:
                print(f"[DEBUG] ⚠️ Low score ({max_score:.4f}) — skipping cache.")

            elif max_score > 0.20:
                # High confidence — evaluate before caching
                print(f"[DEBUG] 🕵️ Evaluating: '{optimized_query}'")
                is_valid = await self.evaluate_answer(
                    query=optimized_query,
                    context=context_text,
                    draft_answer=full_answer_for_cache.strip()
                )
                if is_valid:
                    self._save_to_qdrant_cache(
                        qdrant_client, query_vector,
                        optimized_query, full_answer_for_cache.strip()
                    )
                else:
                    print("[DEBUG] ❌ FAIL — not cached.")

            else:
                # Medium confidence (0.05–0.20) — cache without evaluation
                # Hard for critic to verify raw table chunks, but answer is usually correct
                print(f"[DEBUG] 🟡 Medium score ({max_score:.4f}) — caching without evaluation.")
                self._save_to_qdrant_cache(
                    qdrant_client, query_vector,
                    optimized_query, full_answer_for_cache.strip()
                )

        elif full_answer_for_cache.strip() and not course_data and skip_evaluation:
            # Voice mode: cache directly without evaluation for speed
            max_score = max((doc.score for doc in final_docs), default=0.0)
            if max_score >= 0.05:
                self._save_to_qdrant_cache(
                    qdrant_client, query_vector,
                    optimized_query, full_answer_for_cache.strip()
                )
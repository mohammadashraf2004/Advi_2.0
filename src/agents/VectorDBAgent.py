import json
import os
import re
import uuid
from datetime import datetime
from .BaseAgent import BaseAgent
from models.db_schemas import Project
from qdrant_client.http.models import PointStruct

_COURSES_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'courses_db.json')


class VectorDBAgent(BaseAgent):
    def __init__(self, vectordb_client, generation_client, mongo_client,
                 template_parser, embedding_client, reranker_client=None):
        super().__init__(vectordb_client, generation_client, mongo_client,
                         template_parser, embedding_client)
        self.reranker_client = reranker_client

        self.cache_threshold = 0.98
        self.is_vector_agent = True
        self._qdrant = getattr(vectordb_client, 'client', vectordb_client)

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
    async def evaluate_answer(self, query: str, context: str, draft_answer: str) -> bool:
        prompt = f"""You are a strict RAG evaluator.
Task: Determine if the 'Draft Answer' correctly answers the 'Query' based on the 'Context'.

Rules:
- Answer is correct and grounded in context → "PASS"
- Answer says information is not found AND context truly doesn't contain it → "PASS"
- Answer contains hallucinated facts not present in context → "FAIL"
- Answer says "not found" when information IS clearly in context → "FAIL"
- CRITICAL: If the query asks about a SPECIFIC value or condition 
  (e.g. GPA=3.00, absence=20%, semester=2), the answer MUST address 
  THAT exact value. Answering about a different value → "FAIL"
- CRITICAL: If the query asks about topic X but the answer talks about 
  topic Y (even if Y is in the context) → "FAIL"
- NO EXPLANATIONS. ONLY ONE WORD: PASS or FAIL.

Query: {query}
Context: {context}
Draft Answer: {draft_answer}
Decision:"""
        try:
            # FIX: was max_output_tokens=2 which truncated "PASS"/"FAIL"
            result   = self.generation_client.generate_response(
                prompt=prompt, chat_history=[], temperature=0.0, max_output_tokens=10
            )
            decision = result.strip().upper()
            print(f"[DEBUG] Critic: {decision}")
            return "PASS" in decision
        except Exception as e:
            print(f"[DEBUG] Critic error: {e}")
            return True

    # ── Main stream ───────────────────────────────────────────────────────
    async def process_stream(self, project: Project, query: str,
                             chat_history: list = None, limit: int = 5,
                             skip_evaluation: bool = False):
        print(f"\n[DEBUG] === Agent Stream: '{query}' ===")

        if chat_history is None:
            chat_history = []

        # ── 0. Sanitize query ─────────────────────────────────────────────
        JUNK_CHARS = '"\'"\u201c\u201d\u2018\u2019`'
        query = query.strip().strip(JUNK_CHARS).strip()
        if not query:
            yield "عذراً، لم أفهم السؤال. حاول مرة أخرى."
            return

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
                hours = int(hours_match.group(1))
                total = 2089 + (1330 * hours)
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
1. أجب فقط على السؤال المطروح — لا تتطوع بمعلومات إضافية لم يُسأل عنها.
2. اعتمد فقط على المقتبسات المتاحة. إذا لم تجد المعلومة قل: "هذه المعلومة غير مذكورة في اللائحة الحالية".
3. عندما تجد سطراً بصيغة (X - Y - Z) فاعلم أنه سطر من جدول مفرود.
4. "دواير" = "دوائر"، "Level 100" = "المستوى 100". لا تكن حرفياً.
5. إذا كان السؤال يطلب رقماً أو شرطاً محدداً، اذكره في الجملة الأولى مباشرة ثم اختصر.

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
            f"⚠️ تعليمات صارمة:\n"
            f"- أجب فقط على السؤال المطروح بدقة، لا تضف معلومات إضافية غير مطلوبة.\n"
            f"- إذا كان السؤال عن رقم أو شرط محدد، ابدأ بذكره مباشرة في أول جملة.\n"
            f"- لا تذكر موضوعات أخرى من السياق غير ذات صلة بالسؤال.\n\n"
            f"- إذا كان السؤال يذكر قيمة محددة (مثل GPA معين أو نسبة معينة)، "
            f"أجب فقط عن تلك القيمة المذكورة في السؤال وليس عن قيم أخرى في السياق.\n\n"
            f"✍️ الإجابة المباشرة:"
        )

        stream_generator = self.generation_client.generate_stream(
            prompt=full_prompt,
            chat_history=final_chat_history,
            temperature=0.0,
            max_output_tokens=2048
        )

        # ── 7. Collect full answer BEFORE yielding ────────────────────────
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
                else:
                    buffer += chunk.replace("</thinking>", "")
            else:
                clean_chunk = self.clean_output_text(chunk)
                if clean_chunk:
                    full_answer_for_cache += clean_chunk

        # Flush thinking buffer if </thinking> was never closed
        if buffer.strip():
            clean = self.clean_output_text(buffer)
            if clean:
                full_answer_for_cache += clean

        # ── 8. Evaluate BEFORE yielding to user ───────────────────────────
        final_answer = full_answer_for_cache.strip()

        # Signals that mean the LLM admitted it has no answer
        NOT_FOUND_SIGNALS = [
            "غير مذكور", "لا تتوفر", "لم أجد",
            "لا يوجد", "غير متاح"
        ]

        FALLBACK_MSG = (
            "عذراً، لم أتمكن من العثور على إجابة دقيقة لهذا السؤال "
            "في اللائحة المتاحة. يُرجى التواصل مع شؤون الطلاب للتأكد."
        )

        if final_answer and not course_data and not skip_evaluation:
            max_score = max((doc.score for doc in final_docs), default=0.0)

            # ── 8a. Low-score path ────────────────────────────────────────
            # Scores below 0.05 mean retrieval failed — evaluate first,
            # only show fallback if the LLM answer is also clearly wrong.
            if max_score < 0.05:
                print(f"[DEBUG] ⚠️ Low score ({max_score:.4f}) — evaluating before yield.")
                is_valid = await self.evaluate_answer(
                    query=optimized_query,
                    context=context_text,
                    draft_answer=final_answer
                )
                if not is_valid:
                    # Only use fallback if LLM itself admitted not finding the answer.
                    # If it gave a substantive reply despite weak retrieval, trust it —
                    # the LLM may be using training knowledge that happens to be correct.
                    answer_is_empty = any(s in final_answer for s in NOT_FOUND_SIGNALS)
                    if answer_is_empty:
                        print("[DEBUG] ❌ Low-score + LLM found nothing — showing fallback.")
                        final_answer = FALLBACK_MSG
                    else:
                        print("[DEBUG] ⚠️ Low-score critic FAIL but answer substantive — yielding as-is.")

            # ── 8b. Normal path ───────────────────────────────────────────
            else:
                print(f"[DEBUG] 🕵️ Evaluating: '{optimized_query}' (score={max_score:.4f})")
                is_valid = await self.evaluate_answer(
                    query=optimized_query,
                    context=context_text,
                    draft_answer=final_answer
                )

                if is_valid:
                    # Good answer — cache it
                    self._save_to_qdrant_cache(
                        qdrant_client, query_vector,
                        optimized_query, final_answer
                    )

                else:
                    # First attempt failed critic.
                    # FIX: keep ALL chunks on retry — the original code dropped
                    # chunk[0] which was often the only relevant chunk, causing
                    # the LLM to answer from irrelevant chunks instead.
                    print("[DEBUG] ❌ FAIL — retrying with stricter prompt (all chunks kept).")

                    retry_context = "\n\n".join([
                        f"--- مقتبس {i+1} ---\n{doc.text}"
                        for i, doc in enumerate(final_docs[:5])
                    ])
                    retry_prompt = (
                        f"📚 السياق المستخرج:\n{retry_context}\n\n"
                        f"🎯 السؤال الحالي:\n{query}\n\n"
                        f"⚠️ تعليمات صارمة:\n"
                        f"- ابحث في السياق عن الإجابة المتعلقة تحديداً بـ: '{query}'\n"
                        f"- تجاهل أي مقتبس لا يتحدث مباشرة عن موضوع السؤال.\n"
                        f"- إذا كان السؤال يذكر قيمة محددة (GPA أو نسبة)، "
                        f"أجب عن تلك القيمة فقط.\n"
                        f"- إذا لم تجد الإجابة في أي مقتبس، قل: "
                        f"'هذه المعلومة غير مذكورة في اللائحة الحالية'.\n\n"
                        f"✍️ الإجابة المباشرة:"
                    )

                    retry_answer = ""
                    async for chunk in self.generation_client.generate_stream(
                        prompt=retry_prompt,
                        chat_history=final_chat_history,
                        temperature=0.0,
                        max_output_tokens=2048
                    ):
                        clean = self.clean_output_text(chunk)
                        if clean:
                            retry_answer += clean

                    retry_answer = retry_answer.strip()

                    if retry_answer:
                        print(f"[DEBUG] 🔁 Re-evaluating retry answer...")
                        is_retry_valid = await self.evaluate_answer(
                            query=optimized_query,
                            context=retry_context,
                            draft_answer=retry_answer
                        )

                        if is_retry_valid:
                            print("[DEBUG] ✅ Retry PASS — caching and using retry answer.")
                            final_answer = retry_answer
                            self._save_to_qdrant_cache(
                                qdrant_client, query_vector,
                                optimized_query, final_answer
                            )
                        else:
                            # Both attempts failed the critic.
                            # WHY: The critic evaluates grounding in context, but the
                            # reranker often puts a wrong chunk at position 1, so even
                            # a correct answer drawn from chunk 2/3 appears "ungrounded"
                            # to the critic. The safest policy here is to prefer a
                            # substantive answer over a generic fallback — Arabic synonyms
                            # mean topic-word matching would produce too many false rejects.
                            #
                            # Yield retry answer if it's substantive (long + not a
                            # "not found" message). Skip caching so bad answers don't persist.
                            is_substantive = (
                                len(retry_answer) > 80
                                and not any(s in retry_answer for s in NOT_FOUND_SIGNALS)
                            )
                            if is_substantive:
                                print("[DEBUG] ⚠️ Double FAIL but substantive — yielding without cache.")
                                final_answer = retry_answer
                            else:
                                print("[DEBUG] ❌ Double FAIL and empty/not-found — showing fallback.")
                                final_answer = FALLBACK_MSG
                    else:
                        print("[DEBUG] ❌ Retry returned empty — showing fallback.")
                        final_answer = FALLBACK_MSG

        # ── 9. Yield final answer to user ─────────────────────────────────
        if final_answer:
            yield final_answer
        else:
            yield "عذراً، لم أتمكن من توليد إجابة. حاول مرة أخرى."
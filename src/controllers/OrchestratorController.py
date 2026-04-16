import logging
import json
from datetime import datetime


class Orchestrator:
    def __init__(self, vectordb_client, generation_client, mongo_client,
                 template_parser, embedding_client, reranker_client):
        self.logger            = logging.getLogger(__name__)
        self.generation_client = generation_client
        self.mongo_client      = mongo_client

        from controllers.NLPController import NLPController
        self.nlp_controller = NLPController(
            vectordb_client=vectordb_client,
            generation_client=generation_client,
            embedding_client=embedding_client,
            template_parser=template_parser,
            mongo_client=mongo_client,
            reranker_client=reranker_client
        )

        from agents.VectorDBAgent import VectorDBAgent
        from agents.JobAgent import JobAgent
        from agents.CourseAgent import CourseAgent

        self.vector_agent = VectorDBAgent(
            vectordb_client, generation_client, mongo_client,
            template_parser, embedding_client, reranker_client
        )
        self.job_agent = JobAgent(
            vectordb_client, generation_client, mongo_client,
            template_parser, embedding_client
        )
        self.course_agent = CourseAgent(
            vectordb_client, generation_client, mongo_client,
            template_parser, embedding_client
        )

    # ── Chat history loader ───────────────────────────────────────────────
    async def _get_chat_history(self, project_id):
        try:
            history_data = await self.mongo_client.db["chat_history"].find(
                {"project_id": project_id}
            ).sort("timestamp", -1).limit(3).to_list(length=3)

            chat_history = []
            for msg in reversed(history_data):
                chat_history.append({"role": "user",      "content": msg['query']})
                chat_history.append({"role": "assistant", "content": msg['answer']})
            return chat_history
        except Exception as e:
            self.logger.error(f"Error fetching history: {e}")
            return []

    # ── Main streaming route ──────────────────────────────────────────────
    async def route_query_stream(
        self,
        project,
        query: str,
        limit: int = 5,
        voice_mode: bool = False,
        raw_mode: bool = False
    ):
        """
        route_query_stream — routes query to the correct agent.

        voice_mode=True  : skip the critic evaluator (faster response)
        raw_mode=True    : skip history loading AND rewrite_query entirely.
                           Use this for voice tool calls where Gemini already
                           handles conversation context — prevents the wrong
                           answer appearing in chat due to stale MongoDB turns.
        """

        # ── raw_mode: bypass history + rewrite entirely ───────────────────
        if raw_mode:
            refined_query     = query
            effective_history = []
            intent            = "NEW_TOPIC"
            self.logger.info(f"[raw_mode] Query used as-is: '{query}'")

        # ── normal mode: load history + rewrite query ─────────────────────
        else:
            chat_history = await self._get_chat_history(project.project_id)
            print(f"\n[DEBUG] 💾 Loaded {len(chat_history)//2} previous turns from MongoDB.")

            analysis      = await self.nlp_controller.rewrite_query(query, chat_history)
            intent        = analysis.get("intent", "NEW_TOPIC")
            refined_query = analysis.get("refined_query", query)
            print(f"[DEBUG] 🪄 NLP Analysis Result: {analysis}")

            effective_history = chat_history if intent == "RELATED" else []
            if intent == "NEW_TOPIC":
                self.logger.info("🧹 Topic Drift Detected. Starting fresh turn.")

        # ── Route classification ──────────────────────────────────────────
        classification_prompt = f"""
أنت نظام توجيه (Router) آلي صارم جداً. مهمتك الوحيدة هي تصنيف سؤال الطالب إلى فئة واحدة فقط.
يمنع كتابة أي مبررات. الإجابة كلمة واحدة فقط: (ACADEMIC, JOB, COURSE).

دليل التصنيف:
1. ACADEMIC: لأي شيء يخص "كلية الهندسة" أو "جامعة المنصورة" (مواد دراسية، لوائح، ساعات معتمدة، GPA، جداول).
   ⚠️ أي سؤال فيه كلمة "مقرر" أو "مادة" هو ACADEMIC فوراً.

2. JOB: للأسئلة عن المسار المهني (سوق العمل، وظائف، تدريبات صيفية، Internships، CV).

3. COURSE: للتعلم الذاتي الخارجي فقط (كورسات أونلاين على Coursera أو Udemy أو YouTube).

السؤال: {refined_query}
التصنيف:"""

        category_res = self.generation_client.generate_response(classification_prompt)
        category     = category_res.strip().upper()
        self.logger.info(f"🚦 Route: {category} | Intent: {intent}")

        # ── Select agent ──────────────────────────────────────────────────
        target_agent = self.vector_agent
        if   "JOB"    in category: target_agent = self.job_agent
        elif "COURSE" in category: target_agent = self.course_agent

        full_response = ""

        # ── Stream ────────────────────────────────────────────────────────
        if hasattr(target_agent, 'process_stream'):
            async for chunk in target_agent.process_stream(
                project, refined_query,
                chat_history=effective_history,
                limit=limit,
                skip_evaluation=voice_mode
            ):
                full_response += chunk
                yield chunk
        else:
            res           = await target_agent.process(
                project, refined_query,
                chat_history=effective_history, limit=limit
            )
            full_response = res[0]
            yield full_response

        # ── Save to MongoDB (skip JOB and raw_mode voice turns) ───────────
        # raw_mode voice turns are managed by Gemini's own context —
        # saving them would pollute the text-chat history with voice Q&A
        should_save = (
            full_response.strip()
            and "JOB" not in category
            and not raw_mode          # ✅ don't save voice tool-call turns
        )

        if should_save:
            try:
                await self.mongo_client.db["chat_history"].insert_one({
                    "project_id": project.project_id,
                    "query":      query,
                    "answer":     full_response.strip(),
                    "timestamp":  datetime.now()
                })
                self.logger.info("💾 Turn saved centrally to MongoDB.")
            except Exception as e:
                self.logger.error(f"❌ Error saving turn to MongoDB: {e}")
        elif "JOB" in category:
            self.logger.info("🚫 Skipped saving JOB turn (real-time data).")
        elif raw_mode:
            self.logger.info("🚫 Skipped saving voice tool-call turn.")
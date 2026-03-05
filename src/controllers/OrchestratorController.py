from agents.VectorDBAgent import VectorDBAgent # تأكد من اسم الكلاس عندك
from agents.JobAgent import JobAgent
from agents.CourseAgent import CourseAgent
import logging

class Orchestrator:
    def __init__(self, vectordb_client, generation_client, mongo_client, template_parser, embedding_client):
        self.logger = logging.getLogger(__name__)
        self.generation_client = generation_client
        
        # تمرير كل المتغيرات المطلوبة للـ BaseAgent لتجنب TypeError
        self.vector_agent = VectorDBAgent(
            vectordb_client, generation_client, mongo_client, template_parser, embedding_client
        )
        self.job_agent = JobAgent(
            vectordb_client, generation_client, mongo_client, template_parser, embedding_client
        )
        self.course_agent = CourseAgent(
            vectordb_client, generation_client, mongo_client, template_parser, embedding_client
        )

    async def route_query(self, project, query: str, limit: int = 5):
        """
        تحليل السؤال وتوجيهه للوكيل المناسب
        """
        classification_prompt = f"""
        أنت نظام توجيه (Router) آلي صارم جداً. مهمتك الوحيدة هي تصنيف سؤال الطالب إلى فئة واحدة فقط من الفئات الثلاث المحددة.
        يمنع كتابة أي مبررات أو مقدمات. الإجابة كلمة واحدة فقط باللغة الإنجليزية: (ACADEMIC, JOB, COURSE).

        دليل التصنيف الصارم:
        - ACADEMIC: للأسئلة الأكاديمية والمواد الجامعية (المتطلبات السابقة للمقررات، لوائح الكلية، الساعات المعتمدة، تسجيل المواد، شروط التخرج، الجي بي إيه).
        - JOB: للأسئلة المهنية (سوق العمل، الوظائف، المهارات التقنية المطلوبة للشركات، السيرة الذاتية).
        - COURSE: للتعلم الذاتي الخارجي فقط (البحث عن كورسات أونلاين على Coursera أو Udemy، تعلم مهارة من الصفر خارج الكلية).

        أمثلة حاكمة (يجب القياس عليها):
        السؤال: "ما هو المتطلب السابق لمقرر التعلم العميق؟"
        التصنيف: ACADEMIC

        السؤال: "إيه أفضل كورسات لتعلم الذكاء الاصطناعي؟"
        التصنيف: COURSE

        السؤال: "متطلبات التخرج من قسم حاسبات؟"
        التصنيف: ACADEMIC

        السؤال: {query}
        التصنيف:"""
        
        # استخدم temperature=0.0 لو دالتك بتدعمها
        category_res = self.generation_client.generate_response(classification_prompt)
        category = category_res.strip().upper()

        self.logger.info(f"🚦 Orchestrator routed query to: {category}")

        # 2. التوجيه بناءً على التصنيف
        if "JOB" in category:
            return await self.job_agent.process(project, query)
        elif "COURSE" in category:
            return await self.course_agent.process(project, query)
        else:
            # الافتراضي هو البحث في لائحة الكلية
            return await self.vector_agent.process(project, query, limit=limit)
        
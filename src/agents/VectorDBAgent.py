from .BaseAgent import BaseAgent
from models.db_schemas import Project

class VectorDBAgent(BaseAgent):
    def __init__(self, vectordb_client, generation_client, mongo_client, template_parser, embedding_client):
        super().__init__(vectordb_client, generation_client, mongo_client, template_parser, embedding_client)

        
    async def process(self, project: Project, query: str, limit: int = 10):
        """
        الوكيل المتخصص في استخراج الإجابات من لائحة الكلية الرسمية.
        """
        answer, full_prompt, chat_history = None, None, None
        
        # 1. البحث الهجين (Hybrid Search)
        # استدعاء دالة البحث التي تدمج Qdrant و BM25 و RRF
        retrieved_documents = await self.search(project, query, limit=limit)
        
        if not retrieved_documents or len(retrieved_documents) == 0:
            return "عذراً، لم أجد معلومات متعلقة بهذا السؤال في اللائحة الحالية.", "", []

        # 2. إعداد الـ System Prompt (شخصية الوكيل)
        system_prompt = (
            "أنت المساعد الأكاديمي الرسمي لكلية الهندسة جامعة المنصورة. "
            "مهمتك هي الإجابة بدقة بناءً على نصوص اللائحة المقدمة فقط. "
            "إذا كانت الإجابة تحتوي على أرقام أو مواد قانونية، اذكرها كما هي. "
            "اجعل إجابتك منظمة في نقاط وسهلة القراءة للطالب."
        )

        # 3. بناء السياق من المستندات المسترجعة
        context_text = "\n\n".join([
            f"--- مقتبس {i+1} ---\n{doc.text}" 
            for i, doc in enumerate(retrieved_documents)
        ])

        # 4. بناء الـ Full Prompt لإرساله للـ LLM
        full_prompt = f"""
بناءً على النصوص التالية من لائحة الكلية:
{context_text}

سؤال الطالب: {query}

الإجابة الموثقة:"""

        # 5. تجهيز Chat History وتوليد الرد
        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt, 
                role=self.generation_client.enums.SYSTEM.value
            )
        ]

        # 6. الحصول على الإجابة النهائية من الـ LLM
        answer = self.generation_client.generate_response(
            prompt=full_prompt, 
            chat_history=chat_history
        )

        return answer, full_prompt, chat_history
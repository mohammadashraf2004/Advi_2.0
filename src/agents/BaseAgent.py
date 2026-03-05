import logging
from duckduckgo_search import DDGS
from models.db_schemas import Project
from stores.llm.LLMEnums import DocumentTypeEnum

class BaseAgent:
    def __init__(self, vectordb_client, generation_client, mongo_client, template_parser, embedding_client):
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.mongo_client = mongo_client
        self.template_parser = template_parser
        self.embedding_client = embedding_client
        # إضافة الـ logger لتجنب AttributeError
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_collection_name(self, project_id: str):
        return f"collection_{project_id}".strip()

    async def search(self, project: Project, query: str, limit: int = 5):
        """
        دالة بحث غير متزامنة (Async) متوافقة مع نظام الوكلاء.
        """
        collection_name = self.create_collection_name(project.project_id)
        
        # تحويل النص لمتجه (Embedding)
        # ملاحظة: إذا كان الـ embedding_client يدعم async، أضف await هنا.
        vector = self.embedding_client.embed_text(text=query, document_type=DocumentTypeEnum.QUERY.value)
        
        if not vector or len(vector) == 0:
            return []
        
        try:
            # التحقق من نوع البحث المتاح واستدعاؤه بـ await
            if hasattr(self.vectordb_client, 'search_hybrid'):
                results = self.vectordb_client.search_hybrid(
                    collection_name=collection_name,
                    query=query,
                    vector=vector,
                    limit=limit,
                    vector_weight=0.5,
                    bm25_weight=0.5
                )
            else:
                results = self.vectordb_client.search_by_vector(
                    collection_name=collection_name,
                    vector=vector,
                    limit=limit
                )
            return results if results else []
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []

    async def web_search(self, query: str) -> str:
        """
        البحث في الويب باستخدام DuckDuckGo مع دعم الـ Async.
        """
        try:
            # استخدام DDGS بشكل آمن مع البحث النصي
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=3)]
                
            if not results:
                return "لا توجد نتائج بحث خارجية متاحة حالياً."
            
            formatted_results = "\n".join([
                f"- {r['title']}: {r['body']}\n  الرابط: {r['href']}" 
                for r in results
            ])
            return formatted_results
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            return "حدث خطأ أثناء محاولة الوصول لنتائج البحث الخارجية."
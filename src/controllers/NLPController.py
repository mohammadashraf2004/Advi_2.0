from .BaseController import BaseController
from models.db_schemas import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List
import json
import re
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from models.db_schemas import RetrievalDocument
import logging
import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CacheEntry:
    results: list
    timestamp: float
    hit_count: int = 0

class SemanticCache:
    """
    كاش دلالي للبحث — يحفظ نتائج الاستعلامات المتشابهة دلالياً.
    يستخدم cosine similarity بين embeddings للتحقق من التشابه.
    """
    def __init__(self, embedding_client, similarity_threshold: float = 0.92, ttl_seconds: int = 3600, max_size: int = 200):
        self.embedding_client   = embedding_client
        self.threshold          = similarity_threshold
        self.ttl                = ttl_seconds
        self.max_size           = max_size
        self._store: dict[str, tuple[list, CacheEntry]] = {}  # key → (embedding, entry)
        self.logger             = logging.getLogger(__name__)

    def _cosine_similarity(self, a: list, b: list) -> float:
        import math
        dot  = sum(x * y for x, y in zip(a, b))
        na   = math.sqrt(sum(x*x for x in a))
        nb   = math.sqrt(sum(x*x for x in b))
        return dot / (na * nb + 1e-9)

    def _evict_expired(self):
        now     = time.time()
        expired = [k for k, (_, e) in self._store.items() if now - e.timestamp > self.ttl]
        for k in expired:
            del self._store[k]

    def _evict_lru(self):
        """إزالة الأقل استخداماً عند امتلاء الكاش"""
        if len(self._store) >= self.max_size:
            lru_key = min(self._store, key=lambda k: (self._store[k][1].hit_count, self._store[k][1].timestamp))
            del self._store[lru_key]

    def get(self, query: str, query_embedding: list) -> Optional[list]:
        self._evict_expired()
        for key, (stored_emb, entry) in self._store.items():
            sim = self._cosine_similarity(query_embedding, stored_emb)
            if sim >= self.threshold:
                entry.hit_count += 1
                self.logger.info(f"💨 Cache HIT (sim={sim:.3f}) for: '{query[:50]}'")
                return entry.results
        return None

    def set(self, query: str, query_embedding: list, results: list):
        self._evict_lru()
        cache_key = hashlib.md5(query.encode()).hexdigest()
        self._store[cache_key] = (query_embedding, CacheEntry(results=results, timestamp=time.time()))
        self.logger.info(f"💾 Cache SET for: '{query[:50]}'")

class NLPController:
    def __init__(self, vectordb_client, generation_client, embedding_client, template_parser, mongo_client, reranker_client):
        self.vectordb_client = vectordb_client
        self.embedding_client = embedding_client
        self.mongo_client = mongo_client
        self.generation_client = generation_client
        self.reranker_client = reranker_client 

        self.semantic_cache = SemanticCache(
            embedding_client=embedding_client,
            similarity_threshold=0.92,
            ttl_seconds=3600,
            max_size=200
        )

    def create_collection_name(self, project_id: str):
        return f"collection_{project_id}".strip()
    
    def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return self.vectordb_client.delete_collection(collection_name=collection_name)
    
    def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = self.vectordb_client.get_collection_info(collection_name=collection_name)
        return json.loads(json.dumps(collection_info, default=lambda x: x.__dict__))
    
    def index_into_vector_db(self, project: Project, chunks: List[DataChunk],
                                   chunks_ids: List[int], do_reset: bool = False):
        collection_name = self.create_collection_name(project_id=project.project_id)
        texts = [ c.chunk_text for c in chunks ]
        metadata = [ c.chunk_metadata for c in  chunks]
        vectors = [
            self.embedding_client.embed_text(text=text, document_type=DocumentTypeEnum.DOCUMENT.value)
            for text in texts
        ]
        _ = self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )
        _ = self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            metadata=metadata,
            vectors=vectors,
            record_ids=chunks_ids,
        )
        return True
    
    async def get_chunks_from_db(self, project_id: str):
        collection = self.mongo_client.chunks 
        cursor = collection.find({"project_id": project_id})
        raw_mongo_documents = await cursor.to_list(length=None)
        project_chunks = []
        for doc in raw_mongo_documents:
            chunk = DataChunk(
                chunk_text=doc.get("chunk_text", ""),
                chunk_metadata=doc.get("chunk_metadata", {})
            )
            project_chunks.append(chunk)
        return project_chunks
    
    def normalize_arabic(self, text: str) -> str:
        text = re.sub(r"[إأآا]", "ا", text)
        text = re.sub(r"ى", "ي", text)
        text = re.sub(r"ئ", "ي", text)
        text = re.sub(r"ة", "ه", text)
        text = re.sub(r"[\u064B-\u065F\u0640]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _enrich_query_locally(self, query: str) -> str:
        enriched_query = query
        
        WORKLOAD_KEYWORDS = r'(ساعات|تسجيل|حمل دراسي|عبء|يسجل)'
        GRADE_TABLE_KEYWORDS = r'(تقدير|درجات|حروف|نتيجه|نتيجة)'

        if re.search(GRADE_TABLE_KEYWORDS, query, re.IGNORECASE) and \
        not re.search(WORKLOAD_KEYWORDS, query, re.IGNORECASE):
            enriched_query += " النقاط النسبة المئوية التقدير الحرفي A B C المعدل التراكمي الدلالات الرقمية"

        if re.search(r'(سنة أولى|سنة اولي|سنه اولي|مستوى 100|المستوى الأول|فرقة اعدادي)', query, re.IGNORECASE):
            enriched_query += " المستوى 100 السنة الأولى الخطة الدراسية مقررات إجبارية"
            
        if re.search(r'(ادار|إدار|هيكل|عميد|وكيل|رئيس)', query, re.IGNORECASE):
            enriched_query += " الهيكل الإداري للبرامج المجلس الأكاديمي المدير التنفيذي المنسق العام"
        
        if re.search(r'(level|ليفل|مستوى|فرقة)', query, re.IGNORECASE):
            enriched_query += " المستوى الدراسي الخطة الدراسية المقررات الإجبارية"

        if re.search(r'(اختياري|اختيارية|المواد الاختيارية)', query, re.IGNORECASE):
            enriched_query += " قائمة المقررات الاختيارية التخصصية مستوى 300 و 400"
        
        if re.search(r'(رابط|لينك|موقع|ويب|url|website)', query, re.IGNORECASE):
            enriched_query += " رابط الموقع الإلكتروني المنصة الذكية نظام ابن الهيثم MyU Mansoura"

        return enriched_query
    
    async def rewrite_query(self, query: str, chat_history: list = []) -> dict:
        if not chat_history:
            return {"intent": "NEW_TOPIC", "refined_query": query}

        # ✅ FIX: Rule-based pre-check BEFORE calling the LLM
        # These are the ONLY signals that indicate a follow-up question
        FOLLOWUP_PRONOUNS = [
            'له', 'لها', 'عليه', 'عنها', 'عنه', 'منه', 'منها',
            'فيه', 'فيها', 'ساعاته', 'ساعاتها', 'متطلبه', 'متطلباته',
            'درجاته', 'درجاتها', 'اسمها', 'اسمه', 'كودها', 'كوده',
            'هذا', 'هذه', 'ذلك', 'تلك', 'نفسه', 'نفسها'
        ]
        FOLLOWUP_STARTERS = [
            'وكم', 'وما', 'وهل', 'وأين', 'ومتى', 'وماذا', 'وكيف',
            'وما هو', 'وما هي', 'وهل يمكن', 'وماهو', 'وماهي'
        ]

        query_stripped = query.strip()

        has_pronoun = any(f' {p} ' in f' {query_stripped} ' or
                        query_stripped.endswith(f' {p}') or
                        query_stripped.startswith(f'{p} ')
                        for p in FOLLOWUP_PRONOUNS)

        has_starter = any(query_stripped.startswith(s) for s in FOLLOWUP_STARTERS)

        has_attached_pronoun = bool(re.search(
        r'[\u0600-\u06FF]{3,}(ها|هم|هن|هما|ه)(?:\s|$)',
        query_stripped
    ))

        if not has_pronoun and not has_starter and not has_attached_pronoun:
            print(f"[DEBUG] 🚀 Rule-based NEW_TOPIC (no pronouns/connectors found)")
            return {"intent": "NEW_TOPIC", "refined_query": query}

        # Only reach here if there's a real follow-up signal
        # Then call the LLM to resolve the pronoun
        recent = chat_history[-2:] if len(chat_history) >= 2 else chat_history
        history_text = ""
        for turn in recent:
            role    = "U" if turn.get("role") == "user" else "A"
            content = str(turn.get("content", ""))
            if role == "A" and len(content) > 400:
                content = content[:200] + "..." + content[-100:]
            history_text += f"{role}: {content}\n"

        prompt = f"""Complete this JSON. Output ONLY the JSON object, no other text.

    {{"intent": "RELATED", "refined_query": "the resolved Arabic question with pronoun replaced"}}

    Rules:
    1. The user's question contains a pronoun or connector referring to the previous conversation.
    2. Replace the pronoun with the actual subject from the conversation history.
    3. The refined_query must be a complete, standalone Arabic question.
    4. NEVER change the topic.

    Conversation:
    {history_text.strip()}

    Current question: {query}

    JSON:"""

        try:
            response = self.generation_client.generate_response(prompt, temperature=0.0)

            if not response or not response.strip():
                return {"intent": "NEW_TOPIC", "refined_query": query}

            clean = response.strip().replace("```json", "").replace("```", "").strip()
            start, end = clean.find('{'), clean.rfind('}')

            if start != -1 and end != -1:
                try:
                    result        = json.loads(clean[start:end+1])
                    refined_query = result.get("refined_query", query).strip() or query

                    if len(refined_query) > 120 or '"' in refined_query or 'الضمير' in refined_query:
                        print(f"[WARN] refined_query looks like prose, reverting.")
                        refined_query = query

                    print(f"[DEBUG] 🪄 Result: intent=RELATED, refined='{refined_query}'")
                    return {"intent": "RELATED", "refined_query": refined_query}
                except json.JSONDecodeError:
                    pass

            return {"intent": "RELATED", "refined_query": query}

        except Exception as e:
            print(f"[ERROR] rewrite_query failed: {e}")
            return {"intent": "NEW_TOPIC", "refined_query": query}
        
    async def search_vector_db_collection(self, project: Project, text: str,
                                           limit: int = 10, chat_history: list = []):
        query_embedding = self.embedding_client.embed_text(
            text=text, document_type=DocumentTypeEnum.QUERY.value
        )

        cached = self.semantic_cache.get(text, query_embedding)
        if cached is not None:
            return cached[:limit]

        local_enriched = self._enrich_query_locally(text)
        final_search_query = local_enriched

        print(f"🪄 FINAL QUERY: '{final_search_query}'")

        normalized_query = self.normalize_arabic(final_search_query)
        collection_name  = self.create_collection_name(project_id=project.project_id)
        internal_k       = max(15, limit * 3)

        vector = self.embedding_client.embed_text(
            text=final_search_query,
            document_type=DocumentTypeEnum.QUERY.value
        )
        semantic_results = []
        if vector:
            qdrant_output = self.vectordb_client.search_by_vector(
                collection_name=collection_name,
                vector=vector,
                limit=internal_k
            )
            semantic_results = qdrant_output or []

        cursor = self.mongo_client.chunks.find({"project_id": str(project.project_id)})
        raw_mongo_documents = await cursor.to_list(length=None)

        keyword_results = []
        if raw_mongo_documents:
            chunk_docs = [
                Document(
                    page_content=doc.get("chunk_metadata", {}).get("normalized_text", doc.get("chunk_text", "")),
                    metadata={"original_text": doc.get("chunk_text", "")}
                )
                for doc in raw_mongo_documents
            ]
            bm25_retriever   = BM25Retriever.from_documents(chunk_docs)
            bm25_retriever.k = internal_k
            keyword_results  = bm25_retriever.invoke(normalized_query)

        if not semantic_results and not keyword_results:
            return []

        rrf_scores = {}
        k_penalty  = 60
        for rank, doc in enumerate(semantic_results):
            rrf_scores[doc.text] = rrf_scores.get(doc.text, 0.0) + 1.0 / (rank + 1 + k_penalty)
        for rank, doc in enumerate(keyword_results):
            original_text = doc.metadata.get("original_text", doc.page_content)
            rrf_scores[original_text] = rrf_scores.get(original_text, 0.0) + 1.0 / (rank + 1 + k_penalty)

        sorted_merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        candidates    = [RetrievalDocument(score=score, text=t) for t, score in sorted_merged]

        if hasattr(self, 'reranker_client') and self.reranker_client:
            final_results = self.reranker_client.rerank(query=final_search_query, docs=candidates, top_n=limit)
        else:
            final_results = candidates[:limit]
    
        self.semantic_cache.set(text, query_embedding, final_results)
        return final_results
    

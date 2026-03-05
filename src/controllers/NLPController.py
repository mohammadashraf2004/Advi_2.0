from .BaseController import BaseController
from models.db_schemas import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List
import json
import re
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from models.db_schemas import RetrievalDocument
from agents.VectorDBAgent import VectorDBAgent
from agents.JobAgent import JobAgent
from agents.CourseAgent import CourseAgent
from .OrchestratorController import Orchestrator



class NLPController:
    def __init__(self, vectordb_client, generation_client, embedding_client, template_parser, mongo_client):
        # ... المتغيرات بتاعتك ...
        
        # تهيئة المايسترو هنا مرة واحدة
        self.orchestrator = Orchestrator(
            vectordb_client=vectordb_client,
            generation_client=generation_client,
            mongo_client=mongo_client,
            template_parser=template_parser,
            embedding_client=embedding_client
        )

        self.vector_agent = VectorDBAgent(vectordb_client, generation_client, mongo_client, template_parser,embedding_client)
        self.job_agent = JobAgent(vectordb_client, generation_client, mongo_client, template_parser, embedding_client)
        self.course_agent = CourseAgent(
    vectordb_client=None, # منع الوصول تماماً
    generation_client=generation_client,
    mongo_client=None,
    template_parser=template_parser,
    embedding_client=None
)

    def create_collection_name(self, project_id: str):
        return f"collection_{project_id}".strip()
    
    def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return self.vectordb_client.delete_collection(collection_name=collection_name)
    
    def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = self.vectordb_client.get_collection_info(collection_name=collection_name)

        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        )
    
    def index_into_vector_db(self, project: Project, chunks: List[DataChunk],
                                   chunks_ids: List[int], 
                                   do_reset: bool = False):
        
        # step1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step2: manage items
        texts = [ c.chunk_text for c in chunks ]
        metadata = [ c.chunk_metadata for c in  chunks]
        vectors = [
            self.embedding_client.embed_text(text=text, 
                                             document_type=DocumentTypeEnum.DOCUMENT.value)
            for text in texts
        ]

        # step3: create collection if not exists
        _ = self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )

        # step4: insert into vector db
        _ = self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            metadata=metadata,
            vectors=vectors,
            record_ids=chunks_ids,
        )

        return True
    
    async def get_chunks_from_db(self, project_id: str):
        """
        Fetches the raw text chunks for a given project from MongoDB.
        Uses AsyncIOMotorClient.
        """
        # 1. Access the specific collection (replace 'chunks' with your actual collection name)
        collection = self.mongo_client.chunks 
        
        # 2. Find all documents matching the project_id
        cursor = collection.find({"project_id": project_id})
        
        # 3. Await the cursor to pull the data into memory
        raw_mongo_documents = await cursor.to_list(length=None)
        
        project_chunks = []
        
        # 4. Convert the Mongo dictionaries back into your DataChunk schema
        for doc in raw_mongo_documents:
            chunk = DataChunk(
                chunk_text=doc.get("chunk_text", ""),
                chunk_metadata=doc.get("chunk_metadata", {})
                # Add any other fields your DataChunk schema expects
            )
            project_chunks.append(chunk)
            
        return project_chunks
    
    # Make sure to include 'async' here!
    async def search_vector_db_collection(self, project: Project, text: str, limit: int = 10):

        # Note: 'text' is already normalized before entering this function.

        # 1. Get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # ==========================================
        # A. SEMANTIC SEARCH (Qdrant)
        # ==========================================
        vector = self.embedding_client.embed_text(
            text=text, 
            document_type=DocumentTypeEnum.QUERY.value
        )

        semantic_results = []
        if vector and len(vector) > 0:
            qdrant_output = self.vectordb_client.search_by_vector(
                collection_name=collection_name,
                vector=vector,
                limit=limit
            )
            semantic_results = qdrant_output if qdrant_output else []

        # ==========================================
        # B. KEYWORD SEARCH (BM25)
        # ==========================================
        # Fetch chunks asynchronously from MongoDB using Motor
        project_chunks = await self.get_chunks_from_db(project.project_id)

        keyword_results = []
        if project_chunks:
            # Convert DB chunks to LangChain Documents
            chunk_docs = [Document(page_content=c.chunk_text) for c in project_chunks]

            # Build and execute BM25 Retriever
            bm25_retriever = BM25Retriever.from_documents(chunk_docs)
            bm25_retriever.k = limit
            keyword_results = bm25_retriever.invoke(text)

        # ==========================================
        # C. HYBRID MERGE (Reciprocal Rank Fusion)
        # ==========================================
        if not semantic_results and not keyword_results:
            return False

        rrf_scores = {}
        k_penalty = 60  # Industry standard constant for RRF

        # 1. Score Semantic Results
        for rank, doc in enumerate(semantic_results):
            if doc.text not in rrf_scores:
                rrf_scores[doc.text] = 0.0
            rrf_scores[doc.text] += 1.0 / (rank + 1 + k_penalty)

        # 2. Score Keyword Results
        for rank, doc in enumerate(keyword_results):
            if doc.page_content not in rrf_scores:
                rrf_scores[doc.page_content] = 0.0
            rrf_scores[doc.page_content] += 1.0 / (rank + 1 + k_penalty)

        # 3. Sort combined results by highest RRF score
        sorted_merged_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 4. Format back into your custom RetrievalDocument schema
        final_results = [
            RetrievalDocument(**{
                "score": score,
                "text": text_content
            })
            for text_content, score in sorted_merged_results[:limit]
        ]

        return final_results
    
    async def answer_rag_question(self, project, query: str, limit: int = 5):
        """
        دالة الواجهة: تمرر السؤال للـ Orchestrator ليقوم بالتوجيه
        """
        return await self.orchestrator.route_query(project, query, limit=limit)
from qdrant_client import models, QdrantClient
from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
import logging
from typing import List
from models.db_schemas import RetrievalDocument
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

class QdrantDBProvider(VectorDBInterface):

    def __init__(self, db_client: str, default_vector_size: int = 768,
                                     distance_method: str = None, index_threshold: int=100):

        self.client = None
        self.db_client = db_client
        self.distance_method = None
        self.default_vector_size = default_vector_size

        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = models.Distance.DOT

        self.logger = logging.getLogger('uvicorn')

    def connect(self):
        print(f"DEBUG: Attempting to connect to Qdrant at: {self.db_client}")
        self.client = QdrantClient(url=self.db_client)

        self._ensure_cache_exists()

    def disconnect(self):
        self.client = None

    def is_collection_existed(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name=collection_name)
    
    def list_all_collections(self) -> List:
        return self.client.get_collections()
    
    def get_collection_info(self, collection_name: str) -> dict:
        return self.client.get_collection(collection_name=collection_name)
    
    def delete_collection(self, collection_name: str):
        if self.is_collection_existed(collection_name):
            self.logger.info(f"Deleting collection: {collection_name}")
            return self.client.delete_collection(collection_name=collection_name)
        
    def create_collection(self, collection_name: str, 
                                embedding_size: int,
                                do_reset: bool = False):
        if do_reset:
            _ = self.delete_collection(collection_name=collection_name)
        
        if not self.is_collection_existed(collection_name):
            self.logger.info(f"Creating new Qdrant collection: {collection_name}")
            
            _ = self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=self.distance_method
                )
            )

            return True
        
        return False
    
    def insert_one(self, collection_name: str, text: str, vector: list,
                         metadata: dict = None, 
                         record_id: str = None):
        
        # Note: Ensure you await async methods
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Can not insert new record to non-existed collection: {collection_name}")
            return False
        
        # Wrap in lists and use insert_many to keep the code DRY
        return self.insert_many(
            collection_name=collection_name,
            texts=[text],
            vectors=[vector], 
            metadata=[metadata],
            record_ids=[record_id],
            batch_size=1
        )
    
    def insert_many(self, collection_name: str, texts: list, 
                          vectors: list, metadata: list = None, 
                          record_ids: list = None, batch_size: int = 50):
        
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Can not insert new record to non-existed collection: {collection_name}")
            return False

        if metadata is None:
            metadata = [None] * len(texts)

        # Generate UUIDs if none are provided to prevent Qdrant from silently overwriting data
        if record_ids is None:
            record_ids = [uuid.uuid4().hex for _ in range(len(texts))]

        try:
            for i in range(0, len(texts), batch_size):
                batch_end = i + batch_size

                batch_texts = texts[i:batch_end]
                batch_vectors = vectors[i:batch_end]
                batch_metadata = metadata[i:batch_end]
                batch_records_ids = record_ids[i:batch_end]

                # Map your data to Qdrant's PointStruct
                batch_points = [
                    models.PointStruct(
                        # Fallback to a new UUID if a specific ID in the list is None
                        id=batch_records_ids[x] or uuid.uuid4().hex,
                        vector=batch_vectors[x],
                        payload={
                            "text": batch_texts[x],
                            "metadata": batch_metadata[x]
                        }
                    )
                    for x in range(len(batch_texts))
                ]

                # Upsert into Qdrant
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch_points,
                    wait=True  # CRITICAL: Forces Qdrant to confirm the write before moving on
                )

            self.logger.info(f"Successfully inserted {len(texts)} vectors into {collection_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error while inserting batch into Qdrant: {e}")
            return False
        
    def search_by_vector(self, collection_name: str, vector: list, limit: int = 5):

        results = self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit
        )

        if not results or len(results) == 0:
            return None
        
        return [
            RetrievalDocument(**{
                "score": result.score,
                "text": result.payload["text"],
            })
            for result in results
        ]
    
    def _ensure_cache_exists(self):
        # التأكد من أن الاتصال موجود
        if not self.client:
            return

        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == "semantic_cache" for c in collections)
            
            if not exists:
                self.logger.info("[DEBUG] 🛠️ Creating semantic_cache collection...")
                self.client.create_collection(
                    collection_name="semantic_cache",
                    vectors_config=models.VectorParams(
                        size=3072, # خليها 1536 لـ OpenAI و 768 لـ BGE
                        distance=self.distance_method or models.Distance.COSINE
                    ),
                )
                print("✅ Semantic cache collection created!")
        except Exception as e:
            self.logger.error(f"Failed to ensure cache existence: {e}")
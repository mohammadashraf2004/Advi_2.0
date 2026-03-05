import os
import pickle
import logging
import numpy as np
import faiss
from typing import List, Dict, Optional
from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
from models.db_schemas import RetrievalDocument
import hashlib

# Import BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logging.warning("rank-bm25 not installed. BM25 search will not be available.")


class FAISSProvider(VectorDBInterface):

    def __init__(self, db_path: str, distance_method: str):
        
        self.client = None # Maintained for variable parity with Qdrant
        self.db_path = db_path
        self.distance_method = distance_method
        self.logger = logging.getLogger(__name__)

        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

        # Map Qdrant/Enum distance methods to FAISS metrics
        if distance_method == DistanceMethodEnums.COSINE.value:
            self.faiss_metric = faiss.METRIC_INNER_PRODUCT
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.faiss_metric = faiss.METRIC_INNER_PRODUCT
        else:
            self.faiss_metric = faiss.METRIC_L2  # Default Euclidean

    def _get_index_path(self, collection_name: str) -> str:
        return os.path.join(self.db_path, f"{collection_name}.index")

    def _get_payload_path(self, collection_name: str) -> str:
        return os.path.join(self.db_path, f"{collection_name}.pkl")

    def _get_bm25_path(self, collection_name: str) -> str:
        return os.path.join(self.db_path, f"{collection_name}_bm25.pkl")

    def connect(self):
        # FAISS is file-based, so no persistent connection is needed
        pass

    def disconnect(self):
        pass

    def is_collection_existed(self, collection_name: str) -> bool:
        return os.path.exists(self._get_index_path(collection_name))
    
    def list_all_collections(self) -> List:
        if not os.path.exists(self.db_path):
            return []
        return [f.replace(".index", "") for f in os.listdir(self.db_path) if f.endswith(".index")]
    
    def get_collection_info(self, collection_name: str) -> dict:
        if not self.is_collection_existed(collection_name):
            return None
        
        index = faiss.read_index(self._get_index_path(collection_name))
        
        # Check if BM25 exists
        has_bm25 = os.path.exists(self._get_bm25_path(collection_name))
        
        return {
            "name": collection_name,
            "vectors_count": index.ntotal,
            "embedding_size": index.d,
            "has_bm25_index": has_bm25
        }
    
    def delete_collection(self, collection_name: str):
        if self.is_collection_existed(collection_name):
            index_path = self._get_index_path(collection_name)
            payload_path = self._get_payload_path(collection_name)
            bm25_path = self._get_bm25_path(collection_name)
            
            os.remove(index_path)
            if os.path.exists(payload_path):
                os.remove(payload_path)
            if os.path.exists(bm25_path):
                os.remove(bm25_path)
            return True
        return False
        
    def create_collection(self, collection_name: str, 
                                embedding_size: int,
                                do_reset: bool = False):
        if do_reset:
            _ = self.delete_collection(collection_name=collection_name)
        
        if not self.is_collection_existed(collection_name):
            
            if self.faiss_metric == faiss.METRIC_INNER_PRODUCT:
                index = faiss.IndexFlatIP(embedding_size)
            else:
                index = faiss.IndexFlatL2(embedding_size)
                
            faiss.write_index(index, self._get_index_path(collection_name))
            
            # Create empty payload storage
            with open(self._get_payload_path(collection_name), "wb") as f:
                pickle.dump([], f)

            return True
        
        return False
    
    def insert_one(self, collection_name: str, text: str, vector: list,
                         metadata: dict = None, 
                         record_id: str = None):
        
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Can not insert new record to non-existed collection: {collection_name}")
            return False
        
        # Wrap in lists and use insert_many to keep the code DRY
        return self.insert_many(
            collection_name=collection_name,
            texts=[text],
            vectors=[vector],  # Fixed: was [vectors], should be [vector]
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

        if record_ids is None:
            record_ids = [None] * len(texts)

        index_path = self._get_index_path(collection_name)
        payload_path = self._get_payload_path(collection_name)

        try:
            # Load Index and Payloads
            index = faiss.read_index(index_path)
            with open(payload_path, "rb") as f:
                payloads = pickle.load(f)

            for i in range(0, len(texts), batch_size):
                batch_end = i + batch_size

                batch_texts = texts[i:batch_end]
                batch_vectors = vectors[i:batch_end]
                batch_metadata = metadata[i:batch_end]
                batch_records_ids = record_ids[i:batch_end]

                # Convert to numpy array (FAISS requirement)
                np_vectors = np.array(batch_vectors, dtype=np.float32)

                # CRITICAL FIX: If COSINE similarity is requested, vectors MUST be L2 normalized
                # This ensures document vectors are normalized when stored, matching query normalization
                if self.distance_method == DistanceMethodEnums.COSINE.value:
                    faiss.normalize_L2(np_vectors)
                    self.logger.debug(f"Normalized {len(batch_vectors)} vectors for COSINE similarity")

                # Add to FAISS index
                index.add(np_vectors)

                # Add to Pickled Payloads
                for x in range(len(batch_texts)):
                    payloads.append({
                        "id": batch_records_ids[x],
                        "text": batch_texts[x],
                        "metadata": batch_metadata[x]
                    })

            # Save updates back to disk
            faiss.write_index(index, index_path)
            with open(payload_path, "wb") as f:
                pickle.dump(payloads, f)

            # Build/update BM25 index
            self._build_bm25_index(collection_name, payloads)

            self.logger.info(f"Successfully inserted {len(texts)} vectors into {collection_name}")

        except Exception as e:
            self.logger.error(f"Error while inserting batch: {e}")
            return False

        return True
    
    def _build_bm25_index(self, collection_name: str, payloads: List[Dict]):
        """Build BM25 index from text payloads"""
        if not BM25_AVAILABLE:
            self.logger.warning("BM25 not available, skipping BM25 index build")
            return
        
        try:
            # Extract texts for BM25
            texts = [p.get("text", "") for p in payloads if p.get("text")]
            
            if not texts:
                self.logger.warning(f"No texts to build BM25 index for {collection_name}")
                return
            
            # Simple tokenization (split by whitespace)
            # For Arabic, this works reasonably well, but could be improved with proper tokenization
            tokenized_texts = [text.lower().split() for text in texts]
            
            # Build BM25 index
            bm25 = BM25Okapi(tokenized_texts)
            
            # Save BM25 index
            bm25_data = {
                "bm25": bm25,
                "texts": texts,
                "tokenized_texts": tokenized_texts
            }
            
            bm25_path = self._get_bm25_path(collection_name)
            with open(bm25_path, "wb") as f:
                pickle.dump(bm25_data, f)
            
            self.logger.info(f"Built BM25 index for {collection_name} with {len(texts)} documents")
            
        except Exception as e:
            self.logger.error(f"Error building BM25 index: {e}")
    
    def _load_bm25_index(self, collection_name: str) -> Optional[Dict]:
        """Load BM25 index if it exists"""
        bm25_path = self._get_bm25_path(collection_name)
        
        if not os.path.exists(bm25_path):
            return None
        
        try:
            with open(bm25_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading BM25 index: {e}")
            return None
        
    def search_by_vector(self, collection_name: str, vector: list, limit: int = 5):
        
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Collection {collection_name} does not exist for search")
            return []

        try:
            index = faiss.read_index(self._get_index_path(collection_name))
            with open(self._get_payload_path(collection_name), "rb") as f:
                payloads = pickle.load(f)

            # Log collection stats for debugging
            self.logger.info(f"Searching collection {collection_name}: {index.ntotal} vectors, {len(payloads)} payloads")

            if index.ntotal == 0:
                self.logger.warning(f"Collection {collection_name} is empty")
                return []

            np_vector = np.array([vector], dtype=np.float32)

            # Normalize query vector if using cosine (document vectors are already normalized during insertion)
            if self.distance_method == DistanceMethodEnums.COSINE.value:
                faiss.normalize_L2(np_vector)
                self.logger.debug("Normalized query vector for COSINE similarity")

            distances, indices = index.search(np_vector, limit)

            if len(indices) == 0 or indices[0][0] == -1:
                self.logger.warning(f"No results found in {collection_name}")
                return []
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(payloads):
                    item = payloads[idx]
                    
                    results.append(RetrievalDocument(**{
                        "score": float(distances[0][i]),
                        "text": item.get("text"),
                    }))
                    
            self.logger.info(f"Found {len(results)} results from {collection_name}")
            return results

        except Exception as e:
            self.logger.error(f"Error while searching in {collection_name}: {e}")
            return []
    
    def search_by_text_bm25(self, collection_name: str, query: str, limit: int = 5) -> List[RetrievalDocument]:
        """
        Search using BM25 keyword matching
        """
        if not BM25_AVAILABLE:
            self.logger.error("BM25 not available. Install rank-bm25 package.")
            return []
        
        bm25_data = self._load_bm25_index(collection_name)
        if not bm25_data:
            self.logger.error(f"BM25 index not found for {collection_name}")
            return []
        
        try:
            # Tokenize query
            tokenized_query = query.lower().split()
            
            # Get scores
            bm25 = bm25_data["bm25"]
            scores = bm25.get_scores(tokenized_query)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:limit]
            
            # Build results
            results = []
            payloads = self._load_payloads(collection_name)
            
            for idx in top_indices:
                if idx < len(payloads) and scores[idx] > 0:
                    item = payloads[idx]
                    results.append(RetrievalDocument(**{
                        "score": float(scores[idx]),
                        "text": item.get("text"),
                    }))
            
            self.logger.info(f"BM25 search found {len(results)} results for '{query[:50]}...'")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in BM25 search: {e}")
            return []
    
    def _load_payloads(self, collection_name: str) -> List[Dict]:
        """Load payloads from disk"""
        payload_path = self._get_payload_path(collection_name)
        try:
            with open(payload_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading payloads: {e}")
            return []
    
    def search_hybrid(self, collection_name: str, query: str, vector: list, 
                      limit: int = 5, vector_weight: float = 0.5, 
                      bm25_weight: float = 0.5) -> List[RetrievalDocument]:
        """
        Hybrid search combining FAISS vector search and BM25 keyword search
        Uses Reciprocal Rank Fusion (RRF) for combining results
        
        Args:
            min_score_threshold: Minimum RRF score for a result to be included (0.0-1.0)
        """
        # Get results from both methods
        vector_results = self.search_by_vector(collection_name, vector, limit=limit * 2)
        bm25_results = self.search_by_text_bm25(collection_name, query, limit=limit * 2)
        
        if not vector_results and not bm25_results:
            self.logger.warning(f"No results from either search method for query: '{query[:50]}...'")
            return []
        
        # Reciprocal Rank Fusion (RRF) parameters
        k = 60  # RRF constant
        
        # Create score dictionaries
        vector_scores = {}
        bm25_scores = {}
        
        # Helper function to generate a collision-free key based on the full text
        def generate_key(text: str) -> str:
            if not text:
                return ""
            return hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Rank scores from vector search
        for rank, doc in enumerate(vector_results):
            key = generate_key(doc.text)
            if key:
                vector_scores[key] = {
                    "rank": rank + 1,
                    "score": doc.score,
                    "text": doc.text,
                    "full_doc": doc
                }
        
        # Rank scores from BM25
        for rank, doc in enumerate(bm25_results):
            key = generate_key(doc.text)
            if key:
                bm25_scores[key] = {
                    "rank": rank + 1,
                    "score": doc.score,
                    "text": doc.text,
                    "full_doc": doc
                }
        
        # Combine using RRF
        all_keys = set(vector_scores.keys()) | set(bm25_scores.keys())
        fused_scores = []
        
        for key in all_keys:
            vector_rank = vector_scores.get(key, {}).get("rank", float('inf'))
            bm25_rank = bm25_scores.get(key, {}).get("rank", float('inf'))
            
            # RRF score calculation
            rrf_score = 0.0
            if vector_rank != float('inf'):
                rrf_score += vector_weight * (1.0 / (k + vector_rank))
            if bm25_rank != float('inf'):
                rrf_score += bm25_weight * (1.0 / (k + bm25_rank))
            
            # Safely get the document text from whichever dictionary has it
            text = vector_scores.get(key, {}).get("text") or bm25_scores.get(key, {}).get("text", "")
            
            fused_scores.append({
                "key": key,
                "rrf_score": rrf_score,
                "text": text,
                "vector_rank": vector_rank if vector_rank != float('inf') else None,
                "bm25_rank": bm25_rank if bm25_rank != float('inf') else None,
            })
        
        # Sort by RRF score descending
        fused_scores.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        # Filter by minimum score threshold and apply limit
        
        # Build final results
        results = []
        for item in fused_scores[:limit]:
            results.append(RetrievalDocument(**{
                "score": item["rrf_score"],
                "text": item["text"],
            }))
        
        # Log warning if all results were filtered out due to low scores
        if len(fused_scores) > 0 and len(results) == 0:
            max_score = fused_scores[0]["rrf_score"]
            self.logger.warning(
                f"All {len(fused_scores)} results filtered out due to low scores "
                f"(max: {max_score:.4f}, threshold: {fused_scores}). "
                f"Query may not match any documents: '{query[:50]}...'"
            )
        
        self.logger.info(f"Hybrid search returned {len(results)} results "
                         f"(vector: {len(vector_results)}, bm25: {len(bm25_results)}, "
                         f"filtered from {len(fused_scores)} by threshold {fused_scores})")
        
        return results
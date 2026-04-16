from sentence_transformers import CrossEncoder
from typing import List

class BGERerankerClient:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.reranker = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[any], top_n: int = 5):
        if not docs:
            return []

        pairs = [[query, doc.text] for doc in docs]
        scores = self.reranker.predict(pairs)

        for doc, score in zip(docs, scores):
            # 🌟 السحر هنا: تحويل float32 إلى float عادي باستخدام float()
            doc.score = float(score)

        sorted_docs = sorted(docs, key=lambda x: x.score, reverse=True)
        return sorted_docs[:top_n]
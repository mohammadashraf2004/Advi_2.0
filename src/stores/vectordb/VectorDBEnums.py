from enum import Enum

class VectorDBEnums(Enum):
    QDRANT = "QDRANT"
    FAISS = "FAISS"

class DistanceMethodEnums(Enum):
    COSINE = "cosine"
    DOT = "dot"
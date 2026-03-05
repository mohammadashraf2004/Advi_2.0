from .providers.FAISSProvider import FAISSProvider
from .providers.QDrantDB import QdrantDBProvider
from .VectorDBEnums import VectorDBEnums
from controllers.BaseController import BaseController

class VectorDBProviderFactory:
    def __init__(self, config):
        self.config = config
        self.base_controller = BaseController()

    def create(self, provider: str):
        if provider == VectorDBEnums.QDRANT.value:
            # FIX: Skip get_database_path and use the raw URL string
            qdrant_db_client = self.config.VECTOR_DB_PATH

            return QdrantDBProvider(
                db_client=qdrant_db_client,
                distance_method=self.config.VECTOR_DB_DISTANCE_METHOD,
                default_vector_size=self.config.EMBEDDING_MODEL_SIZE,
            )
            
        if provider == VectorDBEnums.FAISS.value:
            # Leave FAISS exactly as it is, since it actually needs the local folder path
            db_path = self.base_controller.get_database_path(db_name=self.config.VECTOR_DB_PATH)

            return FAISSProvider(
                db_client=db_path,
                distance_method=self.config.VECTOR_DB_DISTANCE_METHOD,
                default_vector_size=self.config.EMBEDDING_MODEL_SIZE,
            )
        return None
        
    
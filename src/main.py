from fastapi import FastAPI
from routes import base, data ,nlp,voice
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.llm.templates.template_parser import TemplateParser
from fastapi.middleware.cors import CORSMiddleware
from controllers.VoiceController import VoiceController

app = FastAPI()


# 2. ADD THIS MIDDLEWARE BLOCK
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your frontend to connect from anywhere during development
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)

async def startup_span():
    settings = get_settings()
    # 1. Store the connection (to close it later)
    app.state.mongo_conn = AsyncIOMotorClient(settings.MONGO_URI)
    
    # 2. Store the specific database object
    app.state.db_client = app.state.mongo_conn[settings.MONGO_DB_NAME]

    llm_provider_factory = LLMProviderFactory(settings)
    vector_db_provider_factory = VectorDBProviderFactory(settings)
    #generation client
    app.state.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
    app.state.generation_client.set_generation_model(model_id =settings.GENERATION_MODEL_ID)
    print(f"✅ LLM Client Initialized: {settings.GENERATION_BACKEND} (Model: {settings.GENERATION_MODEL_ID})")

    #embedding client
    app.state.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
    
    # ⚠️ OVERRIDE THE FACTORY URL: Force the client to point to real OpenAI servers, not Ollama
    app.state.embedding_client.client.base_url = "https://api.openai.com/v1/"
    
    app.state.embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE
    )

    app.state.vectordb_client = vector_db_provider_factory.create(provider=settings.VECTOR_DB_BACKEND)
    app.state.vectordb_client.connect()

    app.state.template_parser = TemplateParser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG,
    )

    app.state.voice_controller = VoiceController(settings)


async def shutdown_span():
    # Properly close the connection
    if hasattr(app.state, "mongo_conn"):
        app.state.mongo_conn.close()
        print("🛑 MongoDB connection closed")

app.on_event("startup")(startup_span)
app.on_event("shutdown")(shutdown_span)


app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)
app.include_router(voice.voice_router)

from fastapi import FastAPI
from routes import base, data, nlp, voice
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.llm.templates.template_parser import TemplateParser
from fastapi.middleware.cors import CORSMiddleware
from controllers.VoiceController import VoiceController
from stores.reranker.RerankerModel import BGERerankerClient
from routes.planner import planner_router
from controllers.OrchestratorController import Orchestrator


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def startup_span():
    settings = get_settings()

    # MongoDB
    app.state.mongo_conn = AsyncIOMotorClient(settings.MONGO_URI)
    app.state.db_client  = app.state.mongo_conn[settings.MONGO_DB_NAME]

    # LLM clients
    llm_factory    = LLMProviderFactory(settings)
    vectordb_factory = VectorDBProviderFactory(settings)

    app.state.generation_client = llm_factory.create(
        provider=settings.GENERATION_BACKEND
    )
    app.state.generation_client.set_generation_model(
        model_id=settings.GENERATION_MODEL_ID
    )
    print(f"✅ LLM Client Initialized: {settings.GENERATION_BACKEND} "
          f"(Model: {settings.GENERATION_MODEL_ID})")

    app.state.embedding_client = llm_factory.create(
        provider=settings.EMBEDDING_BACKEND
    )

    # Override embedding client URL if needed (OpenAI embedding via Ollama base)
    if (hasattr(app.state.embedding_client, "client")
            and hasattr(app.state.embedding_client.client, "base_url")):
        app.state.embedding_client.client.base_url = "https://api.openai.com/v1/"

    app.state.embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE
    )

    # Reranker
    app.state.reranker_client = BGERerankerClient(model_name="BAAI/bge-reranker-base")

    # Vector DB
    app.state.vectordb_client = vectordb_factory.create(
        provider=settings.VECTOR_DB_BACKEND
    )
    app.state.vectordb_client.connect()

    # Template parser
    app.state.template_parser = TemplateParser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG,
    )

    # Voice controller (STT + TTS + Gemini Live)
    app.state.voice_controller = VoiceController(settings)

    # ✅ Orchestrator singleton — must come LAST after all clients are ready
    app.state.orchestrator = Orchestrator(
        vectordb_client=app.state.vectordb_client,
        generation_client=app.state.generation_client,
        mongo_client=app.state.db_client,
        template_parser=app.state.template_parser,
        embedding_client=app.state.embedding_client,
        reranker_client=app.state.reranker_client,
    )
    print("✅ Orchestrator initialized and registered.")


async def shutdown_span():
    if hasattr(app.state, "mongo_conn"):
        app.state.mongo_conn.close()
        print("🛑 MongoDB connection closed")


app.on_event("startup")(startup_span)
app.on_event("shutdown")(shutdown_span)

app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)
app.include_router(voice.voice_router)
app.include_router(planner_router)
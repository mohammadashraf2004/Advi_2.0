from fastapi import FastAPI, APIRouter, status, Request
from fastapi.responses import JSONResponse
from routes.schemas.nlp import PushRequest, SearchRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from controllers import NLPController
from models import ResponseSignal

import logging

logger = logging.getLogger('uvicorn.error')

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1", "nlp"],
)

@nlp_router.post("/index/push/{project_id}")
async def index_project(request: Request, project_id: str, push_request: PushRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.state.db_client
    )

    chunk_model = await ChunkModel.create_instance(
        db_client=request.app.state.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )
    db_client = request.app.state.db_client

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value
            }
        )
    
    nlp_controller = NLPController(
        vectordb_client=request.app.state.vectordb_client,
        generation_client=request.app.state.generation_client,
        embedding_client=request.app.state.embedding_client,
        template_parser=request.app.state.template_parser,
        mongo_client=db_client
    )

    has_records = True
    page_no = 1
    inserted_items_count = 0
    idx = 0

    while has_records:
        page_chunks = await chunk_model.get_poject_chunks(project_id=project.id, page_no=page_no)
        
        if not page_chunks or len(page_chunks) == 0:
            has_records = False
            break

        chunks_ids = list(range(idx, idx + len(page_chunks)))
        idx += len(page_chunks)
        
        # FIX: Only reset the database on the VERY FIRST page. 
        # For all subsequent pages, do_reset MUST be False.
        current_do_reset = push_request.do_reset if page_no == 1 else False
        
        is_inserted = nlp_controller.index_into_vector_db(
            project=project,
            chunks=page_chunks,
            do_reset=current_do_reset,  # <--- FIXED
            chunks_ids=chunks_ids
        )

        if not is_inserted:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.INSERT_INTO_VECTORDB_ERROR.value
                }
            )
        
        # Increment page_no only after processing to track first-page logic properly
        page_no += 1
        inserted_items_count += len(page_chunks)
        
    return JSONResponse(
        content={
            "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
            "inserted_items_count": inserted_items_count
        }
    )

@nlp_router.get("/index/info/{project_id}")
async def get_project_index_info(request: Request, project_id: str):
    
    project_model = await ProjectModel.create_instance(
        db_client=request.app.state.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )
    db_client = request.app.state.db_client

    nlp_controller = NLPController(
        vectordb_client=request.app.state.vectordb_client,
        generation_client=request.app.state.generation_client,
        embedding_client=request.app.state.embedding_client,
        template_parser=request.app.state.template_parser,
        mongo_client=db_client
    )

    collection_info = nlp_controller.get_vector_db_collection_info(project=project)

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_COLLECTION_RETRIEVED.value,
            "collection_info": collection_info
        }
    )

@nlp_router.post("/index/search/{project_id}")
async def search_index(request: Request, project_id: str, search_request: SearchRequest):
    
    project_model = await ProjectModel.create_instance(
        db_client=request.app.state.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )
    
    db_client = request.app.state.db_client

    nlp_controller = NLPController(
        vectordb_client=request.app.state.vectordb_client,
        generation_client=request.app.state.generation_client,
        embedding_client=request.app.state.embedding_client,
        template_parser=request.app.state.template_parser,
        mongo_client=db_client

    )

    results = await nlp_controller.search_vector_db_collection(
        project=project, text=search_request.text, limit=search_request.limit
    )

    if not results:
        return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.VECTORDB_SEARCH_ERROR.value
                }
            )
    
    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
            "results": [ result.dict()  for result in results ]
        }
    )

@nlp_router.post("/index/answer/{project_id}")
async def answer_rag(request: Request, project_id: str, search_request: SearchRequest):
    
    project_model = await ProjectModel.create_instance(
        db_client=request.app.state.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )
    db_client = request.app.state.db_client

    nlp_controller = NLPController(
        vectordb_client=request.app.state.vectordb_client,
        generation_client=request.app.state.generation_client,
        embedding_client=request.app.state.embedding_client,
        template_parser=request.app.state.template_parser,
        mongo_client=db_client
    )

    answer, full_prompt, chat_history = await nlp_controller.answer_rag_question(
        project=project,
        query=search_request.text,
        limit=search_request.limit,
    )

    if not answer:
        return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.RAG_ANSWER_ERROR.value
                }
        )
    
    return JSONResponse(
        content={
            "signal": ResponseSignal.RAG_ANSWER_SUCCESS.value,
            "answer": answer,
            "full_prompt": full_prompt,
            "chat_history": chat_history
        }
    )
    # ==========================================================
    # STEP 1: SUPERVISOR (Routing Logic)
    # ==========================================================
    routing_prompt = f"""
    أنت موجه مهام (Router) للذكاء الاصطناعي الخاص بطلاب هندسة المنصورة. 
    اقرأ سؤال الطالب وحدد من يجب أن يجيب عليه:
    1. "CourseAgent": إذا كان السؤال عن المقررات الدراسية، الساعات المعتمدة، اللوائح، التسجيل، أو تفاصيل المواد.
    2. "JobAgent": إذا كان السؤال عن الوظائف، التدريبات الصيفية، سوق العمل، المهارات، أو التوجيه المهني.
    
    أجب بكلمة واحدة فقط: CourseAgent أو JobAgent.
    
    السؤال: {search_request.text}
    """
    
    # Generate the routing decision
    selected_agent = request.app.state.generation_client.generate_response(
        prompt=routing_prompt, 
        chat_history=[]
    ).strip()

    # ==========================================================
    # STEP 2: AGENT EXECUTION
    # ==========================================================
    # Initialize variables to hold the tuple return values
    answer = None
    full_prompt = None
    chat_history = None

    if "JobAgent" in selected_agent:
        print("💼 Routing to Job Agent...")
        # Pass the individual clients from the app state
        job_agent = JobAgent(
            vectordb_client=request.app.state.vectordb_client,
            embedding_client=request.app.state.embedding_client,
            generation_client=request.app.state.generation_client,
            template_parser=request.app.state.template_parser
        )
        answer, full_prompt, chat_history = job_agent.process(
            project=project, 
            query=search_request.text
        )

    else:
        print("📚 Routing to Course Agent...")
        # Pass the individual clients from the app state
        course_agent = CourseAgent(
            vectordb_client=request.app.state.vectordb_client,
            embedding_client=request.app.state.embedding_client,
            generation_client=request.app.state.generation_client,
            template_parser=request.app.state.template_parser
        )
        answer, full_prompt, chat_history = course_agent.process(
            project=project, 
            query=search_request.text
        )

    # ==========================================================
    # STEP 3: RESPONSE HANDLING
    # ==========================================================
    if not answer:
        return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.RAG_ANSWER_ERROR.value
                }
        )
    
    return JSONResponse(
        content={
            "signal": ResponseSignal.RAG_ANSWER_SUCCESS.value,
            "answer": answer,
            "full_prompt": full_prompt,
            "chat_history": chat_history
        }
    )

import urllib.parse
from fastapi import APIRouter, Request, UploadFile, File, Response, status
from fastapi.responses import JSONResponse
import logging
from helpers.config import get_settings # تأكد من اسم المجلد helpers
from controllers.VoiceController import VoiceController
from controllers.NLPController import NLPController
from models.ProjectModel import ProjectModel

logger = logging.getLogger(__name__)
voice_router = APIRouter(prefix="/api/v1/voice", tags=["Voice"])

@voice_router.post("/chat/{project_id}")
async def single_endpoint_voice_chat(request: Request, project_id: str, file: UploadFile = File(...)):
    app_settings = get_settings()
    voice_controller = VoiceController(settings=app_settings)
    db_client = request.app.state.db_client
    
    nlp_controller = NLPController(
        vectordb_client=request.app.state.vectordb_client,
        generation_client=request.app.state.generation_client,
        embedding_client=request.app.state.embedding_client,
        template_parser=request.app.state.template_parser,
        mongo_client=db_client
    )

    project_model = await ProjectModel.create_instance(db_client=db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    # 1. الاستماع (STT)
    user_question = await voice_controller.transcribe_audio(file)
    if not user_question:
        return JSONResponse(status_code=400, content={"message": "Could not understand audio."})

    # 🌟 2. المسار السريع (Fast Path) للتحيات والوداع
    text_check = user_question.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").lower()
    
    greeting_keywords = ["اهلا", "مرحبا", "سلام عليكم", "عرف نفسك", "انت مين", "مين انت", "بتعمل ايه", "hello", "hi"]
    farewell_keywords = ["سلام", "مع السلامه", "اسكت", "اقفل", "غور", "باي", "goodbye", "bye", "تصبح على خير"]

    if any(word in text_check for word in greeting_keywords):
        llm_answer = "يا أهلاً بيك يا سيدي! أنا ADVI، المرشد الأكاديمي بتاعك في هندسة المنصورة. أنا متدرب على اللائحة وكل تفاصيل الكلية عشان أرد على أي سؤال يخص دراستك، كورساتك، أو حتى سوق العمل. أقدر أساعدك في إيه النهاردة؟"
    
    elif any(word in text_check for word in farewell_keywords):
        llm_answer = "حبيبي يا هندسة! روح شوف مذاكرتك وراك إيه وربنا يوفقك. لو احتجتني في أي استشارة أكاديمية، أنا دايماً هنا.. سلام يا صاحبي!"
    
    else:
        # 3. التفكير العميق للأسئلة الأكاديمية (يتم فقط لو مفيش تحية أو وداع)
        try:
            llm_answer, full_prompt, chat_history = await nlp_controller.answer_rag_question(
                project=project, query=user_question, limit=5
            )
        except Exception as e:
            return JSONResponse(status_code=500, content={"message": str(e)})

    # 4. توليد الصوت بناءً على الإجابة (سواء من المسار السريع أو من الـ NLP)
    audio_bytes = await voice_controller.generate_audio_response(text=llm_answer)

    # 5. تشفير النصوص ووضعها في الـ Headers
    safe_transcription = urllib.parse.quote(user_question)
    safe_answer = urllib.parse.quote(llm_answer)
    
    headers = {
        "X-Transcribed-Text": safe_transcription,
        "X-Answer-Text": safe_answer,
        "Access-Control-Expose-Headers": "X-Transcribed-Text, X-Answer-Text" 
    }

    # 6. إرجاع النتيجة للمتصفح
    return Response(
        content=audio_bytes, 
        media_type="audio/mpeg", 
        headers=headers
    )
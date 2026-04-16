import re
import urllib.parse
import logging
import asyncio
from fastapi import APIRouter, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from helpers.config import get_settings
from models.ProjectModel import ProjectModel

logger       = logging.getLogger(__name__)
voice_router = APIRouter(prefix="/api/v1/voice", tags=["Voice"])


# ── Helpers ───────────────────────────────────────────────────────────────
def _clean_for_tts(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\*{1,2}', '', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'_+', '', text)
    arabic_nums = {
        '0':'صفر','1':'واحد','2':'اتنين','3':'تلاتة','4':'أربعة','5':'خمسة',
        '6':'ستة','7':'سبعة','8':'تمانية','9':'تسعة','10':'عشرة',
        '11':'حداشر','12':'اتناشر','15':'خمستاشر','20':'عشرين',
        '25':'خمسة وعشرين','30':'تلاتين','50':'خمسين','100':'مية',
    }
    for num, word in arabic_nums.items():
        text = re.sub(rf'\b{num}\b', word, text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


async def _collect_stream(async_gen) -> str:
    parts = []
    async for chunk in async_gen:
        if chunk:
            parts.append(chunk)
    return "".join(parts)


def _shorten_for_voice(text: str, generation_client, max_chars: int = 600) -> str:
    if len(text) <= max_chars:
        return text

    voice_prompt = f"""لديك هذه الإجابة التفصيلية التي ستُعرض للطالب على الشاشة:
"{text[:1500]}"

اكتب نصاً منطوقاً قصيراً (جملتان أو ثلاث بالكثير) كأنك ترد في مكالمة هاتفية سريعة.
القواعد الإجبارية:
- لغة مصرية مرحة ("بص يا هندسة"، "الخلاصة يا سيدي")
- ممنوع أي رموز (*, -, #) أو قوائم
- إذا كانت المعلومات كثيرة قل "وباقي التفاصيل قدامك على الشاشة"
- أرقام بالحروف فقط (3 = تلاتة)

النص المنطوق فقط بدون مقدمات:"""

    try:
        short = generation_client.generate_response(
            prompt=voice_prompt,
            chat_history=[],
            temperature=0.3
        )
        return _clean_for_tts(short) if short else text[:max_chars]
    except Exception:
        return text[:max_chars]


# ── HTTP POST — record → transcribe → TTS (fallback) ─────────────────────
@voice_router.post("/chat/{project_id}")
async def voice_chat(request: Request, project_id: str,
                     file: UploadFile = File(...)):

    if await request.is_disconnected():
        return JSONResponse(status_code=499,
                            content={"message": "Client closed the request."})

    voice_controller  = request.app.state.voice_controller
    generation_client = request.app.state.generation_client

    user_question = await voice_controller.transcribe_audio(file)
    if not user_question:
        return JSONResponse(status_code=400,
                            content={"message": "Could not understand audio."})

    logger.info(f"STT result: '{user_question}'")

    text_check = user_question.replace("أ","ا").replace("إ","ا").replace("آ","ا").lower()
    GREETINGS  = ["اهلا","مرحبا","سلام عليكم","عرف نفسك","انت مين","hello","hi"]
    FAREWELLS  = ["مع السلامه","اسكت","اقفل","غور","باي","goodbye","bye"]

    raw_text_answer  = ""
    voice_audio_text = ""

    if any(w in text_check for w in GREETINGS):
        raw_text_answer = voice_audio_text = (
            "يا أهلاً بيك! أنا ADVI، المرشد الأكاديمي بتاعك في هندسة المنصورة. "
            "أقدر أساعدك في اللائحة، الكورسات، أو سوق العمل. قول لي إيه اللي تحب تعرفه!"
        )
    elif any(w in text_check for w in FAREWELLS):
        raw_text_answer = voice_audio_text = (
            "حبيبي يا هندسة! روح شوف مذاكرتك. لو احتجتني، أنا دايماً هنا. سلام!"
        )
    else:
        if not hasattr(request.app.state, 'orchestrator'):
            return JSONResponse(status_code=503,
                                content={"message": "Service not ready."})

        db_client     = request.app.state.db_client
        orchestrator  = request.app.state.orchestrator

        project_model = await ProjectModel.create_instance(db_client=db_client)
        project       = await project_model.get_project_or_create_one(
            project_id=project_id
        )

        try:
            stream          = orchestrator.route_query_stream(
                project=project, query=user_question,
                limit=5, voice_mode=True
            )
            raw_text_answer = await _collect_stream(stream)
        except asyncio.CancelledError:
            logger.warning("Voice request cancelled by client.")
            raise
        except Exception as e:
            logger.error(f"Orchestrator error: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"message": str(e)})

        voice_audio_text = _shorten_for_voice(raw_text_answer, generation_client)
        voice_audio_text = _clean_for_tts(voice_audio_text)

    if not voice_audio_text.strip():
        voice_audio_text = raw_text_answer[:600] or "عذراً، لم أتمكن من توليد رد الآن."

    async def stream_with_disconnect_check():
        try:
            async for chunk in voice_controller.stream_audio_response(
                text=voice_audio_text
            ):
                if await request.is_disconnected():
                    logger.warning("Client disconnected during audio stream.")
                    break
                yield chunk
        except asyncio.CancelledError:
            pass

    MAX_HEADER_CHARS = 800
    headers = {
        "X-Transcribed-Text":      urllib.parse.quote(user_question[:MAX_HEADER_CHARS]),
        "X-Answer-Text":           urllib.parse.quote(raw_text_answer[:MAX_HEADER_CHARS]),
        "X-Answer-Truncated":      "true" if len(raw_text_answer) > MAX_HEADER_CHARS else "false",
        "Access-Control-Expose-Headers": "X-Transcribed-Text, X-Answer-Text, X-Answer-Truncated",
    }

    return StreamingResponse(
        stream_with_disconnect_check(),
        media_type="audio/mpeg",
        headers=headers
    )


# ── WebSocket — Gemini Live bidirectional session ─────────────────────────
@voice_router.websocket("/live/{project_id}")
async def voice_chat_live(websocket: WebSocket, project_id: str):
    await websocket.accept()

    if not hasattr(websocket.app.state, 'voice_controller'):
        logger.error("voice_controller not in app.state")
        await websocket.close(code=1011)
        return

    if not hasattr(websocket.app.state, 'orchestrator'):
        logger.error("orchestrator not in app.state")
        await websocket.close(code=1011)
        return

    voice_controller = websocket.app.state.voice_controller
    orchestrator     = websocket.app.state.orchestrator
    db_client        = websocket.app.state.db_client

    project_model = await ProjectModel.create_instance(db_client=db_client)
    project       = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    system_instruction = (
        "أنت ADVI، مرشد أكاديمي ذكي لكلية الهندسة جامعة المنصورة. "
        "تحدث باللهجة المصرية المرحة والودودة. "
        "مهم جداً: إذا سألك الطالب عن الكورسات أو اللائحة أو أي معلومات تخص الجامعة، "
        "يجب عليك استخدام أداة ask_academic_advisor للبحث عن الإجابة الدقيقة قبل أن ترد. "
        "لا تخترع معلومات أكاديمية من عندك. كن مختصراً ومباشراً."
    )

    try:
        await voice_controller.handle_live_session(
            client_websocket=websocket,
            system_instruction=system_instruction,
            orchestrator=orchestrator,
            project=project
        )
    except WebSocketDisconnect:
        logger.info(f"Student disconnected from Live session (project: {project_id})")
    except Exception as e:
        logger.error(f"Live WebSocket error: {e}", exc_info=True)
        if websocket.client_state.name != 'DISCONNECTED':
            await websocket.close(code=1011)
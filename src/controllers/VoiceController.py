import os
import json
import asyncio
import logging
import time
import aiohttp
from fastapi import WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

INTERRUPT_WORDS = {
    "اسكت", "وقف", "قف", "بس", "كفاية", "سكت", "اوقف",
    "stop", "quiet", "silence", "shut up", "enough"
}

_KEEPALIVE_SILENCE = b'\x00' * 3200


class VoiceController:
    def __init__(self, settings):
        self.settings       = settings
        self.openai_api_key = getattr(settings, "OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        gemini_api_key      = getattr(settings, "GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

        if not self.openai_api_key:
            logger.error("OPENAI_API_KEY missing.")
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY missing.")

        self.gemini_client = genai.Client(
            api_key=gemini_api_key,
            http_options={"api_version": "v1beta"}
        )
        self.gemini_model = "models/gemini-3.1-flash-live-preview"
        self.gemini_voice = "Charon"

        current_dir      = os.path.dirname(os.path.abspath(__file__))
        src_root         = os.path.dirname(current_dir)
        self.cache_dir   = os.path.join(src_root, "assets", "audio_cache")
        self.outputs_dir = os.path.join(src_root, "assets", "outputs")
        os.makedirs(self.cache_dir,   exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

    async def _collect_stream(self, async_gen) -> str:
        parts = []
        async for chunk in async_gen:
            if chunk:
                parts.append(chunk)
        return "".join(parts)

    async def transcribe_audio(self, file) -> str:
        file_content = await file.read()
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        ctx = (
            "جامعة المنصورة، كلية الهندسة، شروط التخرج، الساعات المعتمدة، "
            "اللائحة، المقررات الدراسية، GPA، المتطلبات السابقة، إجباري، اختياري."
        )
        data = aiohttp.FormData()
        data.add_field('file',     file_content, filename=file.filename,
                       content_type=file.content_type or "audio/mpeg")
        data.add_field('model',    'whisper-1')
        data.add_field('language', 'ar')
        data.add_field('prompt',   ctx)
        async with aiohttp.ClientSession() as s:
            async with s.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.openai_api_key}"},
                data=data
            ) as r:
                if r.status != 200:
                    raise RuntimeError(f"Whisper error: {await r.text()}")
                result = await r.json()
                t = result.get("text", "").strip()
                logger.info(f"STT: '{t}'")
                return t

    async def stream_audio_response(self, text: str):
        if not text or not text.strip():
            return
        payload = {"model": "tts-1", "input": text[:4096],
                   "voice": "onyx", "response_format": "mp3"}
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    "https://api.openai.com/v1/audio/speech",
                    headers={"Authorization": f"Bearer {self.openai_api_key}",
                             "Content-Type": "application/json"},
                    json=payload
                ) as r:
                    if r.status != 200:
                        logger.error(f"TTS error: {await r.text()}")
                        return
                    async for chunk in r.content.iter_chunked(4096):
                        if chunk:
                            yield chunk
        except Exception as e:
            logger.error(f"TTS error: {e}")

    async def handle_live_session(
        self,
        client_websocket: WebSocket,
        system_instruction: str,
        orchestrator,
        project
    ):
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(
                parts=[types.Part(text=system_instruction)]
            ),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.gemini_voice
                    )
                )
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            tools=[types.Tool(
                function_declarations=[types.FunctionDeclaration(
                    name="ask_academic_advisor",
                    description="ابحث في قاعدة بيانات هندسة المنصورة للإجابة على أسئلة الطالب.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={"query": types.Schema(
                            type=types.Type.STRING,
                            description="سؤال الطالب الأكاديمي"
                        )},
                        required=["query"]
                    )
                )]
            )]
        )

        try:
            async with self.gemini_client.aio.live.connect(
                model=self.gemini_model, config=config
            ) as session:
                logger.info(f"✅ Gemini Live — voice: {self.gemini_voice}")

                state = {
                    "gemini_speaking": False,
                    "last_audio_ts":   0.0,
                    "client_alive":    True,
                }

                mic_queue = asyncio.Queue(maxsize=200)

                # ── Task 1: Read browser audio ────────────────────────────
                async def read_client_audio():
                    try:
                        while state["client_alive"]:
                            try:
                                audio = await asyncio.wait_for(
                                    client_websocket.receive_bytes(), timeout=2.0
                                )
                                if not mic_queue.full():
                                    await mic_queue.put(audio)
                            except asyncio.TimeoutError:
                                continue
                    except WebSocketDisconnect:
                        logger.info("Browser disconnected.")
                    except Exception as e:
                        logger.error(f"read_client_audio error: {e}")
                    finally:
                        state["client_alive"] = False
                        try: mic_queue.put_nowait(None)
                        except asyncio.QueueFull: pass

                # ── Task 2: Queue → Gemini ────────────────────────────────
                async def send_mic_to_gemini():
                    try:
                        while state["client_alive"]:
                            audio = await mic_queue.get()
                            if audio is None: break
                            if not state["gemini_speaking"]:
                                try:
                                    # ✅ FIX: Use the specific send_realtime_input method and 'audio' parameter
                                    await session.send_realtime_input(
                                        audio=types.Blob(
                                            data=audio,
                                            mime_type="audio/pcm;rate=16000"
                                        )
                                    )
                                except Exception as e:
                                    logger.error(f"Audio drop: {e}")
                                    state["client_alive"] = False
                                    break
                    except asyncio.CancelledError: pass

                # ── Task 3: Keepalive ─────────────────────────────────────
                async def keepalive():
                    try:
                        while state["client_alive"]:
                            await asyncio.sleep(3)
                            if not state["gemini_speaking"] and state["client_alive"]:
                                if not mic_queue.full():
                                    try: mic_queue.put_nowait(_KEEPALIVE_SILENCE)
                                    except asyncio.QueueFull: pass
                    except asyncio.CancelledError: pass

                # ── Task 4: Speaking timeout watcher ──────────────────────
                async def speaking_timeout():
                    try:
                        while state["client_alive"]:
                            await asyncio.sleep(0.5)
                            if (state["gemini_speaking"]
                                    and state["last_audio_ts"] > 0
                                    and time.monotonic() - state["last_audio_ts"] > 2.0):
                                logger.info("⏱️ Auto-reset speaking state")
                                state["gemini_speaking"] = False
                                state["last_audio_ts"]   = 0.0
                                try: await client_websocket.send_text(json.dumps({"type": "turn_complete"}))
                                except Exception: pass
                    except asyncio.CancelledError: pass

                # ── Task 5: Receive from Gemini → send to browser ─────────
                # ── Task 5: Receive from Gemini → send to browser ─────────
                async def receive_from_gemini():
                    try:
                        # ✅ FIX: Outer while loop added!
                        # The Gemini receive iterator naturally exits after tool calls.
                        # We must re-enter it to keep the session alive for the next question.
                        while state["client_alive"]:
                            async for response in session.receive():
                                if not state["client_alive"]: break

                                if response.data:
                                    state["gemini_speaking"] = True
                                    state["last_audio_ts"]   = time.monotonic()
                                    try: await client_websocket.send_bytes(response.data)
                                    except Exception:
                                        state["client_alive"] = False
                                        break

                                if response.server_content:
                                    sc = response.server_content
                                    if (hasattr(sc, 'input_transcription') and sc.input_transcription
                                            and sc.input_transcription.text and not response.tool_call):
                                        txt = sc.input_transcription.text.strip()
                                        logger.info(f"👂 '{txt}'")
                                        if set(txt.lower().split()) & INTERRUPT_WORDS:
                                            state["gemini_speaking"] = False
                                            state["last_audio_ts"]   = 0.0
                                            try: await client_websocket.send_text(json.dumps({"type": "interrupt"}))
                                            except Exception: pass

                                    if hasattr(sc, 'turn_complete') and sc.turn_complete:
                                        state["gemini_speaking"] = False
                                        state["last_audio_ts"]   = 0.0
                                        try: await client_websocket.send_text(json.dumps({"type": "turn_complete"}))
                                        except Exception: pass

                                if response.tool_call:
                                    for fc in response.tool_call.function_calls:
                                        if fc.name == "ask_academic_advisor":
                                            q = fc.args.get("query", "")
                                            state["gemini_speaking"] = True
                                            try: await client_websocket.send_text(json.dumps({"type": "state", "value": "thinking"}))
                                            except Exception: pass

                                            try:
                                                stream = orchestrator.route_query_stream(
                                                    project=project, query=q,
                                                    limit=3, voice_mode=True, raw_mode=True
                                                )
                                                rag_answer = await self._collect_stream(stream)
                                            except Exception as e:
                                                rag_answer = "عذراً، حدث خطأ."

                                            try:
                                                await client_websocket.send_text(json.dumps({
                                                    "type": "qa_pair", "user_text": q, "answer_text": rag_answer
                                                }))
                                            except Exception: pass

                                            try:
                                                await session.send_tool_response(
                                                    function_responses=[
                                                        types.FunctionResponse(
                                                            id=fc.id,
                                                            name=fc.name,
                                                            response={"result": rag_answer}
                                                        )
                                                    ]
                                                )
                                            except Exception as e:
                                                logger.error(f"Tool response error: {e}")
                                                
                            # Log when the iterator completes and we loop back up
                            logger.info("🔄 Gemini stream ended, re-entering receive loop...")

                    except Exception as e:
                        logger.error(f"receive_from_gemini: {e}")
                    finally:
                        state["client_alive"] = False
                        try: mic_queue.put_nowait(None)
                        except asyncio.QueueFull: pass

                all_tasks = [
                    asyncio.create_task(read_client_audio(),  name="read_client"),
                    asyncio.create_task(send_mic_to_gemini(), name="send_mic"),
                    asyncio.create_task(keepalive(),          name="keepalive"),
                    asyncio.create_task(speaking_timeout(),   name="timeout_watcher"),
                    asyncio.create_task(receive_from_gemini(), name="receive_gemini"),
                ]
                try:
                    done, pending = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)
                    for t in pending: t.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                finally:
                    for t in all_tasks:
                        if not t.done(): t.cancel()

        except Exception as e:
            logger.error(f"Gemini Live failed: {e}")
            if client_websocket.client_state.name != 'DISCONNECTED':
                await client_websocket.close(code=1011)
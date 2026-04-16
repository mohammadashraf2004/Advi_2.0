import os
import json
import asyncio
import logging
import aiohttp
import aiofiles
import hashlib
import struct
from fastapi import WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ── Arabic + English interrupt keywords ──────────────────────────────────
INTERRUPT_WORDS = {
    "اسكت", "وقف", "قف", "بس", "كفاية", "سكت", "اوقف",
    "stop", "quiet", "silence", "shut up", "enough"
}


class VoiceController:
    def __init__(self, settings):
        self.settings       = settings
        self.openai_api_key = getattr(settings, "OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        gemini_api_key      = getattr(settings, "GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

        if not self.openai_api_key:
            logger.error("🚨 OPENAI_API_KEY missing — STT/TTS will fail.")
        if not gemini_api_key:
            logger.error("🚨 GEMINI_API_KEY missing — Live sessions will fail.")

        self.gemini_client = genai.Client(
            api_key=gemini_api_key,
            http_options={"api_version": "v1beta"}
        )
        # ✅ FIX 3: Male voice — "Charon" is a deep male voice in Gemini Live
        self.gemini_model = "models/gemini-3.1-flash-live-preview"
        self.gemini_voice = "Charon"   # male voices: Charon, Fenrir, Orus, Puck

        current_dir      = os.path.dirname(os.path.abspath(__file__))
        src_root         = os.path.dirname(current_dir)
        self.cache_dir   = os.path.join(src_root, "assets", "audio_cache")
        self.outputs_dir = os.path.join(src_root, "assets", "outputs")
        os.makedirs(self.cache_dir,   exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

    # ── Collect async generator ───────────────────────────────────────────
    async def _collect_stream(self, async_gen) -> str:
        parts = []
        async for chunk in async_gen:
            if chunk:
                parts.append(chunk)
        return "".join(parts)

    # ── STT via OpenAI Whisper ────────────────────────────────────────────
    async def transcribe_audio(self, file) -> str:
        file_content = await file.read()
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        university_context = (
            "جامعة المنصورة، كلية الهندسة، شروط التخرج، الساعات المعتمدة، "
            "اللائحة، المقررات الدراسية، الجي بي إيه، التدريب الصيفي، "
            "المتطلبات السابقة، تسجيل المواد، إجباري، اختياري."
        )

        url     = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.openai_api_key}"}
        data    = aiohttp.FormData()
        data.add_field('file',     file_content,
                       filename=file.filename,
                       content_type=file.content_type or "audio/mpeg")
        data.add_field('model',    'whisper-1')
        data.add_field('language', 'ar')
        data.add_field('prompt',   university_context)

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=data) as response:
                if response.status != 200:
                    error = await response.text()
                    logger.error(f"Whisper error: {error}")
                    raise RuntimeError(f"OpenAI STT Error: {error}")
                result        = await response.json()
                transcription = result.get("text", "").strip()
                logger.info(f"STT: '{transcription}'")
                return transcription

    # ── TTS streaming via OpenAI (HTTP fallback) ──────────────────────────
    async def stream_audio_response(self, text: str):
        if not text or not text.strip():
            logger.error("stream_audio_response called with empty text")
            return
        if not self.openai_api_key:
            logger.error("OPENAI_API_KEY missing")
            return

        url     = "https://api.openai.com/v1/audio/speech"
        headers = {"Authorization": f"Bearer {self.openai_api_key}", "Content-Type": "application/json"}
        payload = {"model": "tts-1", "input": text[:4096], "voice": "onyx", "response_format": "mp3"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error = await response.text()
                        logger.error(f"TTS error {response.status}: {error}")
                        return
                    async for chunk in response.content.iter_chunked(4096):
                        if chunk:
                            yield chunk
        except Exception as e:
            logger.error(f"TTS error: {e}")

    # ── ✅ FIX 2: Resample PCM to avoid audio noise / frame drops ─────────
    @staticmethod
    def _resample_pcm(data: bytes, src_rate: int, dst_rate: int) -> bytes:
        """
        Simple linear interpolation resampler for int16 mono PCM.
        Gemini returns 24kHz but browsers play best at a consistent rate.
        This also smooths the boundaries between chunks to prevent clicks.
        """
        if src_rate == dst_rate:
            return data

        samples_in  = len(data) // 2
        samples_out = int(samples_in * dst_rate / src_rate)

        src = struct.unpack(f"{samples_in}h", data)
        out = []

        for i in range(samples_out):
            pos   = i * src_rate / dst_rate
            idx   = int(pos)
            frac  = pos - idx
            s0    = src[idx]
            s1    = src[idx + 1] if idx + 1 < samples_in else s0
            sample = int(s0 + frac * (s1 - s0))
            sample = max(-32768, min(32767, sample))
            out.append(sample)

        return struct.pack(f"{samples_out}h", *out)

    # ── Gemini Live WebSocket session ─────────────────────────────────────
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
                        voice_name=self.gemini_voice   # ✅ FIX 3: male voice
                    )
                )
            ),
            # ✅ FIX 4: Enable input transcription so we can detect interrupt words
            input_audio_transcription=types.AudioTranscriptionConfig(),
            tools=[types.Tool(
                function_declarations=[types.FunctionDeclaration(
                    name="ask_academic_advisor",
                    description="ابحث في قاعدة بيانات هندسة المنصورة للإجابة على أسئلة الطالب.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "query": types.Schema(
                                type=types.Type.STRING,
                                description="سؤال الطالب الأكاديمي"
                            )
                        },
                        required=["query"]
                    )
                )]
            )]
        )

        try:
            async with self.gemini_client.aio.live.connect(
                model=self.gemini_model,
                config=config
            ) as session:
                logger.info(f"✅ Gemini Live session — voice: {self.gemini_voice}")

                # Shared flag: when True, drop incoming audio chunks so we
                # don't feed mic back into Gemini while it's speaking
                is_gemini_speaking = False

                # ── Mic → Gemini ──────────────────────────────────────
                async def send_mic_audio():
                    try:
                        while True:
                            audio_bytes = await client_websocket.receive_bytes()

                            # ✅ FIX 4: Don't send mic data while Gemini is speaking
                            # This prevents the model from hearing its own voice
                            # and getting confused / not stopping
                            if not is_gemini_speaking:
                                await session.send_realtime_input(
                                    audio=types.Blob(
                                        data=audio_bytes,
                                        mime_type="audio/pcm;rate=16000"
                                    )
                                )
                    except WebSocketDisconnect:
                        logger.info("Client disconnected from mic.")
                    except Exception as e:
                        logger.error(f"send_mic_audio error: {e}")

                # ── Gemini → Client ───────────────────────────────────
                async def receive_from_gemini():
                    nonlocal is_gemini_speaking
                    try:
                        async for response in session.receive():

                            # ── Audio chunks → forward to browser ────
                            if response.data:
                                is_gemini_speaking = True

                                # ✅ FIX 2: Resample from 24kHz to 22050Hz
                                # to reduce click artifacts at chunk boundaries
                                resampled = self._resample_pcm(response.data, 24000, 22050)
                                await client_websocket.send_bytes(resampled)

                            # ── Server content (transcripts + turn end) ──
                            if response.server_content:
                                sc = response.server_content

                                # User transcript → check for interrupt words
                                if (hasattr(sc, 'input_transcription')
                                        and sc.input_transcription
                                        and sc.input_transcription.text):

                                    transcript = sc.input_transcription.text.strip()
                                    logger.info(f"👂 Heard: '{transcript}'")

                                    # Send transcript to chat
                                    await client_websocket.send_text(json.dumps({
                                        "type": "transcript",
                                        "role": "user",
                                        "text": transcript
                                    }))

                                    # ✅ FIX 4: Detect interrupt keywords
                                    words = set(transcript.lower().split())
                                    if words & INTERRUPT_WORDS:
                                        logger.info("🛑 Interrupt keyword detected")
                                        is_gemini_speaking = False
                                        await client_websocket.send_text(json.dumps({
                                            "type": "interrupt"
                                        }))

                                # ✅ FIX 4: turn_complete → model finished speaking
                                if hasattr(sc, 'turn_complete') and sc.turn_complete:
                                    logger.info("✅ Turn complete — resuming mic")
                                    is_gemini_speaking = False
                                    await client_websocket.send_text(json.dumps({
                                        "type": "turn_complete"
                                    }))

                            # ── Tool call → RAG ───────────────────────
                            if response.tool_call:
                                for fc in response.tool_call.function_calls:
                                    if fc.name == "ask_academic_advisor":
                                        user_query = fc.args.get("query", "")
                                        logger.info(f"🛠️ Tool call: '{user_query}'")

                                        # Stop mic while doing RAG
                                        is_gemini_speaking = True

                                        await client_websocket.send_text(json.dumps({
                                            "type": "state", "value": "thinking"
                                        }))

                                        try:
                                            # ✅ FIX 1: raw_mode=True so Gemini's
                                            # question is used directly without
                                            # rewrite_query touching MongoDB history
                                            stream = orchestrator.route_query_stream(
                                                project=project,
                                                query=user_query,
                                                limit=3,
                                                voice_mode=True,
                                                raw_mode=True
                                            )
                                            rag_answer = await self._collect_stream(stream)
                                        except Exception as e:
                                            logger.error(f"RAG error: {e}")
                                            rag_answer = "عذراً، حدث خطأ أثناء البحث."

                                        # Send text answer to chat display
                                        await client_websocket.send_text(json.dumps({
                                            "type": "answer",
                                            "text": rag_answer
                                        }))

                                        # Return RAG result to Gemini to speak
                                        await session.send(
                                            input=types.LiveClientToolResponse(
                                                function_responses=[types.FunctionResponse(
                                                    id=fc.id,
                                                    name=fc.name,
                                                    response={"result": rag_answer}
                                                )]
                                            )
                                        )

                    except Exception as e:
                        logger.error(f"receive_from_gemini error: {e}")

                await asyncio.gather(send_mic_audio(), receive_from_gemini())

        except Exception as e:
            logger.error(f"Failed to establish Gemini Live session: {e}")
            if client_websocket.client_state.name != 'DISCONNECTED':
                await client_websocket.close(code=1011)
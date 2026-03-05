import os
import hashlib
import logging
import time
import aiohttp
import aiofiles
from fastapi import UploadFile

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceController:
    def __init__(self, settings):
        self.settings = settings
        
        # API Keys & Paths
        self.openai_api_key = getattr(settings, "OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        self.elevenlabs_api_key = getattr(settings, "ELEVENLABS_API_KEY", os.getenv("ELEVENLABS_API_KEY"))
        self.elevenlabs_voice_id = getattr(settings, "ELEVENLABS_VOICE_ID", os.getenv("ELEVENLABS_VOICE_ID"))
        
        if not self.openai_api_key:
            logger.error("🚨 OPENAI_API_KEY is missing! Transcription and Generation will fail.")

        # Path configurations
        current_dir = os.path.dirname(os.path.abspath(__file__)) # src/controllers
        self.src_root = os.path.dirname(current_dir) # src
        
        # Audio Cache Setup
        self.cache_dir = os.path.join(self.src_root, "assets", "audio_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Outputs Directory
        self.outputs_dir = os.path.join(self.src_root, "assets", "outputs")
        os.makedirs(self.outputs_dir, exist_ok=True)

    async def transcribe_audio(self, file: UploadFile) -> str:
        """Transcribes audio using OpenAI Whisper API directly."""
        file_content = await file.read()
        
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        logger.info("🎧 Transcribing via OpenAI API...")
        
        # ⏱️ START TIMER
        stt_start_time = time.perf_counter()

        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.openai_api_key}"}
        
        # السياق الأكاديمي لمساعدة OpenAI على فهم مصطلحات الكلية
        university_context = (
            "جامعة المنصورة، كلية الهندسة، شروط التخرج، الساعات المعتمدة، "
            "اللائحة، المقررات الدراسية، قسم ميكاترونيكس، حاسبات، الجي بي إيه، "
            "التدريب الصيفي، المتطلبات السابقة، تسجيل المواد، إجباري، اختياري."
        )

        # safe_filename = file.filename if "." in file.filename else "audio_record.webm"
        
        data = aiohttp.FormData()
        data.add_field('file', file_content, filename=file.filename, content_type=file.content_type or "audio/mpeg")
        data.add_field('model', 'whisper-1')
        data.add_field('language', 'ar') 
        data.add_field('prompt', university_context)

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=data) as response:
                if response.status != 200:
                    error_detail = await response.text()
                    logger.error(f"❌ OpenAI API Error: {error_detail}")
                    raise RuntimeError(f"OpenAI STT Error: {error_detail}")
                
                result = await response.json()
                transcription = result.get("text", "").strip()
                
                # ⏱️ END TIMER
                stt_duration = time.perf_counter() - stt_start_time
                
                logger.info(f"⏱️ OpenAI Transcribed in: {stt_duration:.2f} seconds")
                logger.info(f"🗣️ Transcription Result: {transcription}")
                
                return transcription
                
    async def _text_to_speech_openai(self, text: str) -> bytes:
        """Generates audio using OpenAI's ultra-fast tts-1 model."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        # You can change "onyx" to: alloy, echo, fable, nova, or shimmer
        payload = {
            "model": "tts-1",
            "input": text,
            "voice": "onyx", 
            "response_format": "wav" 
        }

        # ⏱️ START OPENAI TIMER
        start_time = time.perf_counter()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_detail = await response.text()
                    raise RuntimeError(f"OpenAI TTS Error: {error_detail}")
                
                # Download the audio bytes directly
                audio_data = await response.read()
                
        # ⏱️ END OPENAI TIMER
        duration = time.perf_counter() - start_time
        logger.info(f"⏱️ OpenAI tts-1 Generated Audio in: {duration:.2f} seconds")
        
        return audio_data
    
    async def _text_to_speech_elevenlabs(self, text: str):
        """Internal async generator for ElevenLabs API."""
        if not self.elevenlabs_voice_id:
            raise ValueError("ELEVENLABS_VOICE_ID is not set.")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.elevenlabs_voice_id}/stream"
        headers = {
            "Accept": "audio/mpeg",
            "xi-api-key": self.elevenlabs_api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2", 
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_detail = await response.text()
                    raise RuntimeError(f"ElevenLabs TTS Error: {error_detail}")
                
                async for chunk in response.content.iter_chunked(1024):
                    yield chunk

    async def generate_audio_response(self, text: str) -> bytes:
        """
        Generates audio using OpenAI TTS-1 for high-speed testing.
        """
        if not text:
            return b""

        logger.info(f"🚨 TEXT RECEIVED BY VOICE CONTROLLER: {text}")

        # 1. Check cache first (Saves money on repeated questions!)
        file_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{file_hash}.audio")

        if os.path.exists(cache_path):
            logger.info("Serving audio from cache...")
            async with aiofiles.open(cache_path, "rb") as f:
                return await f.read()

        # 2. Call OpenAI API directly
        logger.info("🎙️ Requesting audio from OpenAI tts-1...")
        try:
            final_audio = await self._text_to_speech_openai(text)
        except Exception as e:
            logger.error(f"❌ OpenAI TTS Failed: {e}")
            return b""

        # 3. Final verification and Caching
        if not final_audio:
            raise RuntimeError("Voice generation failed.")

        async with aiofiles.open(cache_path, "wb") as f:
            await f.write(final_audio)

        return final_audio

    async def stream_audio_response(self, text: str):
        """
        تقوم ببث الصوت كمقاطع (Chunks) بمجرد توليدها لتقليل وقت الانتظار.
        """
        
        # هنستخدم OpenAI كمثال للبث المباشر، وممكن تستخدم ElevenLabs بنفس الطريقة
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "tts-1",
            "input": text,
            "voice": "onyx",
            "response_format": "mp3" # mp3 أفضل وأسرع في البث من wav
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    raise RuntimeError("TTS Streaming Failed")
                
                # إرسال الصوت كقطرات (Chunks) فور وصولها!
                async for chunk in response.content.iter_chunked(4096):
                    yield chunk
    
    

            # def _generate_via_subprocess():
            #     try:
            #         # 🛠️ THE FIX: Force strict UTF-8 encoding so Arabic doesn't get corrupted in WSL
            #         custom_env = os.environ.copy()
            #         custom_env["PYTHONIOENCODING"] = "utf-8"
            #         custom_env["LANG"] = "C.UTF-8"

            #         # Run the isolated environment
            #         result = subprocess.run(
            #             command, 
            #             capture_output=True, 
            #             text=True, 
            #             check=True,
            #             encoding="utf-8", # Lock the input/output to UTF-8
            #             env=custom_env    # Inject the safe environment variables
            #         )
                    
            #         # If successful, read the bytes from the generated file
            #         with open(output_path, "rb") as f:
            #             audio_data = f.read()
                    
            #         # Clean up the temp file
            #         os.remove(output_path)
            #         return audio_data
            #     except subprocess.CalledProcessError as e:
            #         logger.error(f"❌ Voice Engine Subprocess Crashed!\n{e.stderr}")
            #         return b""
 
            # logger.info("🎙️ Starting Da7ee7 Voice Generation...")
            
            # # ⏱️ START XTTS TIMER
            # tts_start_time = time.perf_counter()

            # # Run subprocess in a threadpool so FastAPI isn't blocked
            # final_audio = await run_in_threadpool(_generate_via_subprocess)

            # # ⏱️ END XTTS TIMER
            # tts_duration = time.perf_counter() - tts_start_time
            # logger.info(f"⏱️ XTTS Subprocess Generated Audio in: {tts_duration:.2f} seconds")

            # # 3. Fallback to ElevenLabs if local failed
            # if not final_audio and self.elevenlabs_api_key:
            #     logger.info("Falling back to ElevenLabs TTS...")
            #     try:
            #         audio_generator = self._text_to_speech_elevenlabs(text)
            #         async for chunk in audio_generator:
            #             final_audio += chunk
            #     except Exception as e:
            #         logger.error(f"ElevenLabs Fallback failed: {e}")

            # # 4. Final verification and Caching
            # if not final_audio:
            #     raise RuntimeError("Voice generation failed: No TTS method succeeded.")

            # async with aiofiles.open(cache_path, "wb") as f:
            #     await f.write(final_audio)

            # return final_audio
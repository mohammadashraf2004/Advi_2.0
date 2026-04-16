from ..LLMinterface import LLMInterface
import google.generativeai as genai
import logging

class GeminiProvider(LLMInterface):

    def __init__(self, api_key: str, api_url: str=None,
                       default_input_max_characters: int=1000,
                       default_generation_max_output_tokens: int=1000,
                       default_generation_temperature: float=0.1):
        
        self.api_key = api_key
        # Gemini عادة بيستخدم الـ API Key مباشرة ومبيحتاجش base_url إلا لو شغال بـ Vertex AI
        genai.configure(api_key=self.api_key)

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        # أمثلة: 'gemini-1.5-pro' أو 'gemini-1.5-flash'
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        # مثال: 'models/embedding-001'
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def _prepare_gemini_messages(self, chat_history: list):
        """
        دالة مساعدة لتحويل الـ Chat History من شكل OpenAI المعتاد لشكل Gemini.
        وتستخرج الـ System Prompt لو موجود عشان يتباصى للموديل كـ system_instruction.
        """
        system_instruction = None
        gemini_messages = []

        for msg in chat_history:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")

            # 1. استخراج الـ System Prompt
            if role == "system":
                system_instruction = content
            else:
                # 2. تحويل 'assistant' لـ 'model' وتجهيز هيكل Gemini
                gemini_role = "model" if role == "assistant" else "user"
                gemini_messages.append({
                    "role": gemini_role,
                    "parts": [content]
                })
        
        return system_instruction, gemini_messages

    def generate_response(self, prompt: str, chat_history: list=None, max_output_tokens: int=None,
                            temperature: float = None):
        
        if chat_history is None:
            chat_history = []

        if not self.generation_model_id:
            self.logger.error("Generation model for Gemini was not set")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        # أخذ نسخة من الهيستوري وإضافة سؤال المستخدم الحالي
        messages_with_prompt = list(chat_history)
        messages_with_prompt.append(
            self.construct_prompt(prompt=prompt, role="user")
        )

        # تحويل الداتا لهيكل جيميناي
        system_instruction, formatted_messages = self._prepare_gemini_messages(messages_with_prompt)

        try:
            # تجهيز الموديل وإضافة الدستور (System Prompt) لو متاح
            model = genai.GenerativeModel(
                model_name=self.generation_model_id,
                system_instruction=system_instruction
            )

            # تجهيز الإعدادات (Config)
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )

            response = model.generate_content(
                formatted_messages,
                generation_config=generation_config
            )

            return response.text

        except Exception as e:
            self.logger.error(f"Error while generating text with Gemini: {e}")
            return None

    # ==========================================
    # 🌟 الدالة الخاصة بالـ Streaming (Asynchronous)
    # ==========================================
    async def generate_stream(self, prompt: str, chat_history: list=None, max_output_tokens: int=None,
                            temperature: float = None):
        
        if chat_history is None:
            chat_history = []

        if not self.generation_model_id:
            self.logger.error("Generation model for Gemini was not set")
            return

        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        messages_with_prompt = list(chat_history)
        messages_with_prompt.append(
            self.construct_prompt(prompt=prompt, role="user")
        )

        system_instruction, formatted_messages = self._prepare_gemini_messages(messages_with_prompt)

        try:
            model = genai.GenerativeModel(
                model_name=self.generation_model_id,
                system_instruction=system_instruction
            )

            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )

            # 🌟 استخدام generate_content_async مع stream=True
            response = await model.generate_content_async(
                formatted_messages,
                generation_config=generation_config,
                stream=True
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            self.logger.error(f"Error while streaming text with Gemini: {e}")
            yield f"[ERROR] حدث خطأ أثناء الاتصال بخوادم Gemini"

    def embed_text(self, text: str, document_type: str = None):
        
        if not self.embedding_model_id:
            self.logger.error("Embedding model for Gemini was not set")
            return None
        
        try:
            # تحديد نوع التضمين بناءً على document_type
            task_type = "RETRIEVAL_QUERY" if document_type == "query" else "RETRIEVAL_DOCUMENT"

            response = genai.embed_content(
                model=self.embedding_model_id,
                content=text,
                task_type=task_type
            )

            if not response or 'embedding' not in response:
                self.logger.error("Error while embedding text with Gemini")
                return None

            return response['embedding']

        except Exception as e:
            self.logger.error(f"Error during Gemini embedding: {e}")
            return None

    def construct_prompt(self, prompt: str, role: str):
        # بنحتفظ بنفس الهيكل عشان نفضل متوافقين مع الكود الأقدم، 
        # ودالة _prepare_gemini_messages هي اللي هتهندل التحويل الداخلي لجيميناي
        return {
            "role": role,
            "content": self.process_text(prompt)
        }
from ..LLMinterface import LLMInterface
from ..LLMEnums import OpenAIEnums
from openai import OpenAI, AsyncOpenAI # 🌟 تم إضافة AsyncOpenAI هنا
import logging

class OpenAIProvider(LLMInterface):

    def __init__(self, api_key: str, api_url: str=None,
                       default_input_max_characters: int=1000,
                       default_generation_max_output_tokens: int=1000,
                       default_generation_temperature: float=0.1):
        
        self.api_key = api_key
        self.api_url = api_url

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        # العميل العادي (Synchronous) للطلبات العادية
        self.client = OpenAI(
            api_key = self.api_key,
            base_url = self.api_url if self.api_url and len(self.api_url) else None
        )

        # 🌟 العميل الجديد (Asynchronous) لطلبات الـ Streaming
        self.async_client = AsyncOpenAI(
            api_key = self.api_key,
            base_url = self.api_url if self.api_url and len(self.api_url) else None
        )

        self.enums = OpenAIEnums
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def generate_response(self, prompt: str, chat_history: list=None, max_output_tokens: int=None,
                            temperature: float = None):
        
        if chat_history is None:
            chat_history = []

        if not self.client:
            self.logger.error("OpenAI client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for OpenAI was not set")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        chat_history.append(
            self.construct_prompt(prompt=prompt, role=OpenAIEnums.USER.value)
        )

        response = self.client.chat.completions.create(
            model = self.generation_model_id,
            messages = chat_history,
            max_tokens = max_output_tokens,
            temperature = temperature
        )

        if not response or not response.choices or len(response.choices) == 0 or not response.choices[0].message:
            self.logger.error("Error while generating text with OpenAI")
            return None

        return response.choices[0].message.content

   # ==========================================
    # 🌟 الدالة الجديدة الخاصة بالـ Streaming
    # ==========================================
    async def generate_stream(self, prompt: str, user_query: str = None, chat_history: list = None, 
                              max_output_tokens: int = None, temperature: float = None):
        """
        دعم الـ Streaming مع التعامل الذكي مع الـ RAG و الـ History.
        - prompt: بيمثل الـ System Instructions أو السياق (Chunks).
        - user_query: السؤال الحالي للطالب.
        """
        if chat_history is None:
            chat_history = []

        # 🛠️ FIX 1: Yield an error string so the UI doesn't freeze with a blank response
        if not self.async_client or not self.generation_model_id:
            self.logger.error("OpenAI client or model ID is missing")
            yield "[ERROR] إعدادات الخادم غير مكتملة (Missing API Client)."
            return

        max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temperature = temperature or self.default_generation_temperature

        # 🧠 ترتيب الرسايل بشكل احترافي للـ RAG
        messages = []
        
        if user_query:
            # 🛠️ FIX 2: Prevent "Double System Prompt" if history already has one
            has_system_prompt = any(msg.get("role") == OpenAIEnums.SYSTEM.value for msg in chat_history)
            
            if not has_system_prompt:
                messages.append(self.construct_prompt(prompt=prompt, role=OpenAIEnums.SYSTEM.value))
            
            messages.extend(chat_history) # التاريخ في النص
            messages.append(self.construct_prompt(prompt=user_query, role=OpenAIEnums.USER.value)) # السؤال في الآخر
        else:
            # التوافق مع الكود القديم: السؤال أو الـ prompt المدمج يكون هو الـ User Message
            messages.extend(chat_history)
            messages.append(self.construct_prompt(prompt=prompt, role=OpenAIEnums.USER.value))

        try:
            response = await self.async_client.chat.completions.create(
                model=self.generation_model_id,
                messages=messages,
                max_tokens=max_output_tokens,
                temperature=temperature,
                stream=True
            )

            async for chunk in response:
                if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            self.logger.error(f"Error while streaming text with OpenAI: {e}")
            yield f"\n[ERROR] حدث خطأ أثناء الاتصال بالخادم: {str(e)}"


    def embed_text(self, text: str, document_type: str = None):
        if not self.client:
            self.logger.error("OpenAI client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for OpenAI was not set")
            return None
        
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model_id,
                input=text,
            )

            if not response or not response.data or len(response.data) == 0 or not response.data[0].embedding:
                self.logger.error("Error while embedding text with OpenAI: Empty Response")
                return None

            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"OpenAI Embedding API failed: {e}")
            return None

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            # افتراض أن process_text دالة موجودة بتنظف النص قبل الإرسال
            "content": self.process_text(prompt) 
        }
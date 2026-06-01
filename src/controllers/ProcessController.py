from .BaseController import BaseController
from .ProjectController import ProjectController
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from models.enums.ProcessingEnums import ProcessingEnums
from typing import List
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .ChunkerController import process_file_content_semantic

# 🌟 أضفنا مكتبة مايكروسوفت الجديدة
from markitdown import MarkItDown 
import re
import tiktoken

# (تم إيقاف استيراد Document من Langchain لمنع التضارب مع الكلاس الخاص بك)
@dataclass
class Document:
    page_content: str
    metadata: dict

# ==========================================
# 🌟 محول مخصص لوثائق الوورد (Custom Loader)
# ==========================================
class MarkItDownDocxLoader:
    """
    كلاس يعمل كـ Wrapper ليحاكي سلوك LangChain Loaders.
    يحتوي على دالة load() تقوم بقراءة الوورد وإرجاع قائمة من الـ Documents.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        md = MarkItDown()
        result = md.convert(self.file_path)
        # إرجاع النص بصيغة Markdown مع الاحتفاظ بمسار الملف في الميتاداتا
        return [Document(page_content=result.text_content, metadata={"source": self.file_path})]


class ProcessController(BaseController):

    def __init__(self, project_id: str):
        super().__init__()

        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)

    def get_file_extension(self, file_id: str):
        return os.path.splitext(file_id)[-1]

    def get_file_loader(self, file_id: str):

        file_ext = self.get_file_extension(file_id=file_id)
        file_path = os.path.join(
            self.project_path,
            file_id
        )

        if not os.path.exists(file_path):
            return None

        if file_ext == ProcessingEnums.TXT.value:
            return TextLoader(file_path, encoding="utf-8")

        if file_ext == ProcessingEnums.PDF.value:
            return PyMuPDFLoader(file_path)
            
        if file_ext == ProcessingEnums.DOCX.value:
            # 🌟 استخدام المحول الذكي الجديد للجداول
            return MarkItDownDocxLoader(file_path)
        
        return None

    def get_file_content(self, file_id: str):

        loader = self.get_file_loader(file_id=file_id)
        if loader:
            return loader.load()

        return None
    
    def normalize_arabic(self, text):
        if not text:
            return ""
        text = re.sub(r"[إأآا]", "ا", text)
        text = re.sub(r"ى", "ي", text)
        text = re.sub(r"ئ", "ي", text)
        text = re.sub(r"ة", "ه", text)
        # 🌟 استخدام Unicode Range لضمان إزالة كل التشكيل والتطويل بدون أخطاء مسافات
        text = re.sub(r"[\u064B-\u065F\u0640]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def process_file_content(
        self,
        file_content: list,
        file_id: str,
        chunk_size: int = 400,
        overlap_size: int = 50
    ):
        return process_file_content_semantic(
            file_content=file_content,
            file_id=file_id,
            max_tokens_per_chunk=chunk_size,
            overlap_tokens=overlap_size,
        )
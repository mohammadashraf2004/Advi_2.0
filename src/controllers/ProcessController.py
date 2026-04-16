from .BaseController import BaseController
from .ProjectController import ProjectController
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from models.enums.ProcessingEnums import ProcessingEnums
from typing import List
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        chunk_size: int = 500,
        overlap_size: int = 80
    ):

        enc = tiktoken.get_encoding("cl100k_base")

        # ---------------------------
        # 1. تنظيف النص
        # ---------------------------
        def clean_text(text):
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            return text.strip()

        # ---------------------------
        # 2. تحويل الجداول إلى نص
        # ---------------------------
        def convert_table_to_text(text):
            if "|" not in text:
                return text

            lines = text.split("\n")
            new_lines = []

            for line in lines:
                if "|" in line:
                    cells = [c.strip() for c in line.split("|") if c.strip()]
                    if len(cells) > 1:
                        new_lines.append("تفاصيل السطر: " + " - ".join(cells))
                else:
                    new_lines.append(line)

            return "\n".join(new_lines)

        # ---------------------------
        # 3. استخراج عنوان المقطع
        # ---------------------------
        def extract_title(text):
            match = re.search(r'\*\*•\s*(.*?)\*\*', text)
            if match:
                return match.group(1).strip()
            return "general"

        # ---------------------------
        # 4. تحديد القسم (Metadata)
        # ---------------------------
        def detect_section(text):
            if "الهيكل الإداري" in text:
                return "administrative_structure"
            elif "الساعات المعتمدة" in text:
                return "credit_hours"
            elif "القيد" in text or "التحويل" in text:
                return "enrollment"
            return "general"

        # ---------------------------
        # تجهيز النصوص
        # ---------------------------
        file_content_texts = []
        file_content_metadata = []

        for rec in file_content:
            text = clean_text(rec.page_content)
            text = convert_table_to_text(text)

            file_content_texts.append(text)
            file_content_metadata.append(rec.metadata)

        # ---------------------------
        # 5. Chunking أذكى
        # ---------------------------
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            separators=[
                "\n\n**•",  # أهم فصل حسب العناوين
                "\n\n",
                "\n",
                "؟",
                "!",
                ".",
                " "
            ],
            length_function=lambda text: len(enc.encode(text))
        )

        chunks = text_splitter.create_documents(
            texts=file_content_texts,
            metadatas=file_content_metadata
        )

        print(f"[DEBUG] Enhancing {len(chunks)} chunks with semantic structure...")

        # ---------------------------
        # 6. تحسين كل Chunk
        # ---------------------------
        for chunk in chunks:
            text = chunk.page_content

            # استخراج العنوان
            title = extract_title(text)

            # إضافة bilingual + context
            enhanced_text = f"""
    العنوان: {title}
    Title: {title}

    المحتوى:
    {text}
    """.strip()

            chunk.page_content = enhanced_text

            # -------------------
            # Metadata
            # -------------------
            chunk.metadata["file_id"] = file_id
            chunk.metadata["section"] = detect_section(text)

            # has_table
            chunk.metadata["has_table"] = "|" in text or "جدول" in text

            # has_url
            chunk.metadata["has_url"] = bool(
                re.search(r'(http|www\.|edu\.eg|myu\.mans)', text, re.IGNORECASE)
            )

            # normalized text (BM25)
            chunk.metadata["normalized_text"] = self.normalize_arabic(text)

        return chunks
    
    # def process_contextual_splitter(self, texts: List[str], metadatas: List[dict], chunk_size: int) -> List[Document]:
    #     chunks = []
        
    #     # نمر على كل صفحة/مستند مع الميتاداتا الخاصة به لعدم ضياعها
    #     for text, base_metadata in zip(texts, metadatas):
    #         lines = text.split('\n')
            
    #         current_heading = "معلومات عامة" # العنوان الافتراضي
    #         current_chunk_text = ""
            
    #         for line in lines:
    #             line = line.strip()
    #             if not line:
    #                 continue
                    
    #             # اكتشاف عناوين المقررات أو مواد اللائحة
    #             # مثال: "✅ CSE 351 – التعلم العميق" أو "مادة [۱]:"
    #             if line.startswith("✅") or line.startswith("مادة [") or (len(line) < 60 and not ":" in line and "هندسة" in line):
    #                 current_heading = line.replace("✅", "").strip()
    #                 continue # نحتفظ به كعنوان ولا نجعله Chunk منفصل
                    
    #             # حقن العنوان داخل السطر ليحتفظ بالسياق!
    #             enriched_line = f"[{current_heading}] {line}\n"
                
    #             # إذا كان السطر الجديد سيتخطى الحد الأقصى لحجم القطعة (chunk_size)
    #             if len(current_chunk_text) + len(enriched_line) > chunk_size:
    #                 if current_chunk_text.strip():
    #                     # دمج الميتاداتا الأصلية مع عنوان المادة الحالي
    #                     new_metadata = base_metadata.copy()
    #                     new_metadata["heading"] = current_heading
                        
    #                     chunks.append(Document(
    #                         page_content=current_chunk_text.strip(),
    #                         metadata=new_metadata
    #                     ))
    #                 # بدء قطعة جديدة
    #                 current_chunk_text = enriched_line
    #             else:
    #                 # إضافة السطر للقطعة الحالية
    #                 current_chunk_text += enriched_line
                    
    #         # حفظ الجزء الأخير المتبقي في نهاية الملف
    #         if current_chunk_text.strip():
    #             new_metadata = base_metadata.copy()
    #             new_metadata["heading"] = current_heading
                
    #             chunks.append(Document(
    #                 page_content=current_chunk_text.strip(),
    #                 metadata=new_metadata
    #             ))

    #     return chunks


    
    

    # def process_simpler_splitter(self, texts: List[str], metadatas: List[dict], chunk_size: int,overlap_size: int = 200, splitter_tag: str="\n"):
    #     chunks = []

    #     # 1. Zip texts and metadatas together to preserve document-level metadata
    #     for text, metadata in zip(texts, metadatas):
            
    #         # Split by splitter_tag and filter out empty strings
    #         lines = [line.strip() for line in text.split(splitter_tag) if len(line.strip()) > 0]
    #         current_chunk = ""

    #         for line in lines:
    #             # 2. Check if adding the new line will exceed the chunk size limit
    #             if current_chunk and (len(current_chunk) + len(line) + len(splitter_tag) > chunk_size):
    #                 # Save the current chunk before it gets too big
    #                 chunks.append(Document(
    #                     page_content=current_chunk.strip(),
    #                     metadata=metadata.copy() 
    #                 ))
    #                 current_chunk = "" # Reset
                
    #             # Now add the line to the current chunk
    #             current_chunk += line + splitter_tag
                
    #             # 3. Edge Case: What if a single line is larger than the chunk_size itself?
    #             # We append it immediately so it doesn't bleed into the next iteration.
    #             if len(current_chunk) >= chunk_size:
    #                 chunks.append(Document(
    #                     page_content=current_chunk.strip(),
    #                     metadata=metadata.copy()
    #                 ))
    #                 current_chunk = ""

    #         # 4. Catch any remaining text at the end of the document (Fixing the >= 0 bug)
    #         if len(current_chunk.strip()) > 0:
    #             chunks.append(Document(
    #                 page_content=current_chunk.strip(),
    #                 metadata=metadata.copy()
    #             ))

    #     return chunks

    

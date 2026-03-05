from .BaseController import BaseController
from .ProjectController import ProjectController
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from models.enums.ProcessingEnums import ProcessingEnums
from typing import List
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
# Assuming you are using LangChain's Document class
from langchain_core.documents import Document 
from langchain_community.document_loaders import Docx2txtLoader
import re
import tiktoken

@dataclass
class Document:
    page_content: str
    metadata: dict

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
            return Docx2txtLoader(file_path)
        
        return None

    def get_file_content(self, file_id: str):

        loader = self.get_file_loader(file_id=file_id)
        if loader:
            return loader.load()

        return None
    
    def normalize_arabic(self,text):
        text = re.sub(r"[إأآا]", "ا", text)
        text = re.sub(r"ى", "ي", text)
        text = re.sub(r"ئ", "ي", text)
        text = re.sub(r"ة", "ه", text)
        text = re.sub(r""" ّ|َ|ً|ُ|ٌ|ِ|ٍ|ْ|ـ""", '', text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def process_file_content(self, file_content: list, file_id: str,
                            chunk_size: int = 500,
                            overlap_size: int = 75):

        # Use same tokenizer as the embedding / LLM model
        enc = tiktoken.get_encoding("cl100k_base")

        # Normalize Arabic text
        file_content_texts = [
            self.normalize_arabic(rec.page_content.strip())
            for rec in file_content
        ]

        file_content_metadata = [
            rec.metadata
            for rec in file_content
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,              # measured in TOKENS now
            chunk_overlap=overlap_size,         # 10–20% recommended
            separators=[
                "\n\n",
                "\n",
                "؟",
                "!",
                "؛",
                "،",
                ". ",
                " "
            ],
            length_function=lambda text: len(enc.encode(text))
        )

        chunks = text_splitter.create_documents(
            texts=file_content_texts,
            metadatas=file_content_metadata
        )

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


    
    

    # def process_simpler_splitter(self, texts: List[str], metadatas: List[dict], chunk_size: int, splitter_tag: str="\n"):
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

    

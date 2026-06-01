"""
semantic_chunker.py
-------------------
Professional semantic chunker for Arabic academic regulation documents.
Splits by actual document sections (** • headers) instead of character count,
so each chunk contains exactly ONE topic — no keyword hacks needed.

Drop-in replacement for the process_file_content method in ProcessController.
"""

import re
import tiktoken
from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    page_content: str
    metadata: dict


# ── Section metadata map ──────────────────────────────────────────────────────
# Maps Arabic section title keywords → semantic category for filtering/routing.
# Add more entries as the document grows.
SECTION_CATEGORY_MAP = [
    (r"نظام الدراسة|الدرجات العلمية",              "study_system"),
    (r"معيار الساعة المعتمدة",                      "credit_hour_standard"),
    (r"الإدارة التنفيذية|التحول الرقمي|هيكل إداري", "administrative_structure"),
    (r"شروط القيد|التحويل|إعادة القيد|قواعد القيد",  "enrollment_transfer"),
    (r"معادلة التقديرات|تقدير الدرجات|الدرجات والنقاط", "grading_table"),
    (r"تقديرات لا تدخل|رموز التقدير",               "grade_symbols"),
    (r"حساب المعدلات|المعدل الفصلي|المعدل التراكمي", "gpa_calculation"),
    (r"تقديرات الخريجين|مرتبة الشرف|شهادة التفوق",   "honors_graduation"),
    (r"المتطلبات السابقة|Prerequisite",              "prerequisites"),
    (r"مدة الدراسة|التقويم الأكاديمي|الفصول الدراسية", "academic_calendar"),
    (r"الحضور|الغياب|المواظبة|الإنذارات الأكاديمية",  "attendance_warnings"),
    (r"العبء الدراسي|الحد الأقصى للساعات",           "credit_load"),
    (r"التسجيل|الإضافة والحذف|الانسحاب",             "registration"),
    (r"إعادة التسجيل|الرسوب",                        "course_repeat"),
    (r"التدريب|الميداني|العملي",                      "training"),
    (r"المشاريع|مشروع التخرج",                       "graduation_project"),
    (r"نظام التقييم|توزيع الدرجات|شروط النجاح",       "assessment_system"),
    (r"المسار الأكاديمي|الإنذار الأكاديمي|حالات الفصل|Academic Standing", "academic_standing"),
    (r"التفوق|مرتبة الشرف",                          "honors"),
    (r"تصنيف المستويات|Classification",              "student_classification"),
    (r"الاستماع|التحسين|نظام التحسين",               "course_improvement"),
    (r"الحالات الخاصة|غير المكتمل|التظلمات",         "special_cases"),
    (r"الإدارة الإلكترونية|الأطر القانونية",          "legal_framework"),
    (r"متطلبات التخرج|توزيع الساعات",               "graduation_requirements"),
    (r"المقررات الاختيارية",                          "elective_courses"),
    (r"خطة التخرج|التخرج المعجل",                    "graduation_plan"),
    (r"المرشد الأكاديمي",                             "academic_advisor"),
    (r"الفصل الصيفي|استراتيجية الفصل الصيفي",        "summer_semester"),
    (r"المسار الحرج",                                "critical_path"),
    (r"الرؤية|الرسالة|أهداف البرنامج|مواصفات الخريج", "program_objectives"),
    (r"تعريف البرنامج|مميزات البرنامج",               "program_definition"),
    (r"كفاءات الخريج|NARS",                          "graduate_competencies"),
    (r"توزيع الساعات المعتمدة|متطلبات الكلية|متطلبات الجامعة", "credit_distribution"),
    (r"الشراكات الأكاديمية|الأقسام المشاركة",         "academic_partnerships"),
    (r"خوارزمية|Algorithm for RAG",                  "rag_algorithm"),
    (r"نموذج الخطة الدراسية",                         "study_plan_model"),
    (r"منظومة المشاريع",                              "projects_system"),
    (r"هيكل العبء الدراسي",                          "weekly_load_structure"),
    (r"تأثير المعدل التراكمي على العبء",              "gpa_load_impact"),
]


def detect_category(title: str) -> str:
    """Map a section title to a semantic category."""
    for pattern, category in SECTION_CATEGORY_MAP:
        if re.search(pattern, title, re.IGNORECASE):
            return category
    return "general"


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for BM25 indexing."""
    if not text:
        return ""
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى",      "ي", text)
    text = re.sub(r"ئ",      "ي", text)
    text = re.sub(r"ة",      "ه", text)
    text = re.sub(r"[\u064B-\u065F\u0640]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def convert_table_to_text(text: str) -> str:
    """Convert markdown pipe tables to readable Arabic rows."""
    if "|" not in text:
        return text
    lines    = text.split("\n")
    new_lines = []
    for line in lines:
        if "|" in line:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            # skip pure separator lines like | --- | --- |
            if cells and not all(re.match(r"^-+$", c) for c in cells):
                new_lines.append("تفاصيل السطر: " + " - ".join(cells))
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


def split_into_sections(raw_text: str) -> List[dict]:
    """
    Split the document on `**•` section headers.
    Returns list of {"title": str, "body": str}.
    Each section covers exactly one academic rule/topic.
    """
    # Pattern: **• Title**: or **• Title** (with optional trailing **)
    header_pattern = re.compile(
        r'(?=\*\*•\s)',   # lookahead so we keep the header in the split
        re.MULTILINE
    )

    parts = header_pattern.split(raw_text)
    sections = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Extract the title from the first line
        title_match = re.match(
            r'\*\*•\s*(.*?)(?:\*\*\s*:|\*\*\s*\n|\*\*:)',
            part,
            re.DOTALL
        )
        if title_match:
            raw_title = title_match.group(1)
            # Clean bold markers and trailing colons from title
            title = re.sub(r'\*+', '', raw_title).strip().rstrip(":")
        else:
            # Fallback: take first non-empty line
            first_line = next((l for l in part.split("\n") if l.strip()), "")
            title = re.sub(r'\*+', '', first_line).strip()[:80] or "general"

        body = part  # keep full text including title line for context
        sections.append({"title": title, "body": body})

    return sections


def process_file_content_semantic(
    file_content: list,
    file_id: str,
    max_tokens_per_chunk: int = 400,
    overlap_tokens: int = 50,
) -> List[Document]:
    """
    Professional semantic chunker.

    Strategy:
    1. Split document on **• section headers — one section = one topic.
    2. If a section is small enough (≤ max_tokens_per_chunk), keep it as one chunk.
    3. If a section is large (e.g. a big table), split it further by
       sub-headers or paragraphs — but NEVER across section boundaries.
    4. Each chunk gets rich metadata: title, category, has_table, has_url,
       normalized_text for BM25.

    This eliminates the need for any keyword-based query enrichment because
    each chunk is semantically coherent and topically pure.
    """
    enc = tiktoken.get_encoding("cl100k_base")

    def token_len(text: str) -> int:
        return len(enc.encode(text))

    def split_large_section(title: str, body: str) -> List[str]:
        """
        Split a section that exceeds max_tokens_per_chunk.
        Tries paragraph boundaries first, then sentence boundaries.
        Always prepends the section title to each sub-chunk for context.
        """
        sub_chunks = []
        # Try splitting on double newlines (paragraphs)
        paragraphs = re.split(r'\n{2,}', body)

        current   = f"[{title}]\n"
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            candidate = current + para + "\n\n"
            if token_len(candidate) <= max_tokens_per_chunk:
                current = candidate
            else:
                if current.strip() and current != f"[{title}]\n":
                    sub_chunks.append(current.strip())
                # If single paragraph is still too large, split by sentence
                if token_len(para) > max_tokens_per_chunk:
                    sentences = re.split(r'(?<=[.؟!])\s+', para)
                    current   = f"[{title}]\n"
                    for sent in sentences:
                        candidate = current + sent + " "
                        if token_len(candidate) <= max_tokens_per_chunk:
                            current = candidate
                        else:
                            if current.strip() and current != f"[{title}]\n":
                                sub_chunks.append(current.strip())
                            current = f"[{title}]\n" + sent + " "
                    if current.strip() and current != f"[{title}]\n":
                        sub_chunks.append(current.strip())
                    current = f"[{title}]\n"
                else:
                    current = f"[{title}]\n" + para + "\n\n"

        if current.strip() and current != f"[{title}]\n":
            sub_chunks.append(current.strip())

        return sub_chunks if sub_chunks else [body[:2000]]  # safety fallback

    all_chunks: List[Document] = []

    for rec in file_content:
        raw_text = rec.page_content
        source   = rec.metadata.get("source", "")

        # Convert tables before splitting
        raw_text = convert_table_to_text(raw_text)

        sections = split_into_sections(raw_text)
        print(f"[INFO] Found {len(sections)} semantic sections in '{source}'")

        for section in sections:
            title    = section["title"]
            body     = section["body"]
            category = detect_category(title)

            # Clean body: remove markdown bold markers but keep structure
            body_clean = re.sub(r'\*{2,}', '', body).strip()
            body_clean = re.sub(r'[ \t]+', ' ', body_clean)
            body_clean = re.sub(r'\n{3,}', '\n\n', body_clean)

            # Decide: one chunk or split?
            if token_len(body_clean) <= max_tokens_per_chunk:
                text_chunks = [body_clean]
            else:
                text_chunks = split_large_section(title, body_clean)

            for i, chunk_text in enumerate(text_chunks):
                # Build the enhanced chunk text with bilingual header
                suffix = f" (جزء {i+1})" if len(text_chunks) > 1 else ""
                enhanced = (
                    f"العنوان: {title}{suffix}\n"
                    f"Title: {title}{suffix}\n\n"
                    f"الموضوع: {title}. "       # ← adds title as readable sentence
                    f"الفئة: {category}.\n\n"   # ← adds category keyword
                    f"المحتوى:\n{chunk_text}"
                )

                metadata = {
                    "source":          source,
                    "file_id":         file_id,
                    "section_title":   title,
                    "section":         category,        # replaces old detect_section()
                    "part_index":      i,
                    "total_parts":     len(text_chunks),
                    "has_table":       "تفاصيل السطر" in chunk_text or "جدول" in chunk_text,
                    "has_url":         bool(re.search(r'(http|www\.|edu\.eg|myu\.mans)', chunk_text, re.IGNORECASE)),
                    "token_count":     token_len(chunk_text),
                    "normalized_text": normalize_arabic(chunk_text),
                }

                all_chunks.append(Document(
                    page_content=enhanced,
                    metadata=metadata,
                ))

    print(f"[INFO] Total semantic chunks produced: {len(all_chunks)}")
    for chunk in all_chunks[:5]:  # preview first 5
        title = chunk.metadata["section_title"]
        cat   = chunk.metadata["section"]
        toks  = chunk.metadata["token_count"]
        print(f"  → [{cat}] '{title[:60]}' ({toks} tokens)")

    return all_chunks


# ── Drop-in replacement ───────────────────────────────────────────────────────
# In ProcessController, replace process_file_content with this:
#
#   from semantic_chunker import process_file_content_semantic
#
#   def process_file_content(self, file_content, file_id,
#                             chunk_size=400, overlap_size=50):
#       return process_file_content_semantic(
#           file_content=file_content,
#           file_id=file_id,
#           max_tokens_per_chunk=chunk_size,
#           overlap_tokens=overlap_size,
#       )
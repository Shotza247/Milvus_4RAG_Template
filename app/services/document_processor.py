import os
import re
import uuid
from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.services.interfaces import DocumentChunk


class DocumentProcessor:
    """
    Handles file parsing, text extraction, normalization, and chunking.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize whitespace and strip illegal characters."""
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def extract_text_from_file(file_path: str, content_type: Optional[str] = None) -> str:
        """Extract plain text from various file formats."""
        ext = os.path.splitext(file_path)[-1].lower()

        if ext in [".txt", ".md", ".json", ".csv", ".log"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        elif ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                pages_text = [page.extract_text() or "" for page in reader.pages]
                return "\n\n".join(pages_text)
            except ImportError:
                logger.warning("pypdf not installed. Falling back to binary read.")
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()

        elif ext == ".docx":
            try:
                import docx
                doc = docx.Document(file_path)
                return "\n".join([p.text for p in doc.paragraphs if p.text])
            except ImportError:
                logger.warning("python-docx not installed.")
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()

        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    @classmethod
    def chunk_text(
        cls,
        text: str,
        doc_id: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        strategy: str = "recursive",
    ) -> List[DocumentChunk]:
        """
        Splits text into chunks using character/word sliding windows or paragraph breaks.
        """
        c_size = chunk_size or settings.CHUNK_SIZE
        c_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        d_id = doc_id or str(uuid.uuid4())
        meta = metadata or {}
        cleaned = cls.clean_text(text)

        if not cleaned:
            return []

        chunks: List[DocumentChunk] = []

        if strategy == "paragraph":
            paragraphs = cleaned.split("\n\n")
            current_chunk = ""
            idx = 0
            for p in paragraphs:
                if len(current_chunk) + len(p) <= c_size:
                    current_chunk += ("\n\n" + p if current_chunk else p)
                else:
                    if current_chunk:
                        chunk_meta = meta.copy()
                        chunk_meta["chunk_index"] = idx
                        chunks.append(
                            DocumentChunk(
                                chunk_id=f"{d_id}_{idx}",
                                doc_id=d_id,
                                text=current_chunk,
                                metadata=chunk_meta,
                            )
                        )
                        idx += 1
                    current_chunk = p
            if current_chunk:
                chunk_meta = meta.copy()
                chunk_meta["chunk_index"] = idx
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{d_id}_{idx}",
                        doc_id=d_id,
                        text=current_chunk,
                        metadata=chunk_meta,
                    )
                )

        else:
            # Recursive sliding window by characters/words
            step = max(1, c_size - c_overlap)
            idx = 0
            for i in range(0, len(cleaned), step):
                chunk_str = cleaned[i : i + c_size].strip()
                if chunk_str:
                    chunk_meta = meta.copy()
                    chunk_meta["chunk_index"] = idx
                    chunk_meta["start_char"] = i
                    chunk_meta["end_char"] = i + len(chunk_str)
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{d_id}_{idx}",
                            doc_id=d_id,
                            text=chunk_str,
                            metadata=chunk_meta,
                        )
                    )
                    idx += 1

        return chunks

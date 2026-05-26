import re
from pathlib import Path


def parse_pdf(file_bytes: bytes) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def parse_docx(file_bytes: bytes) -> str:
    import io
    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def parse_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def parse_document(filename: str, file_bytes: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_bytes)
    elif ext == ".docx":
        return parse_docx(file_bytes)
    elif ext == ".txt":
        return parse_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks of ~chunk_size words.
    Each consecutive chunk shares `overlap` words with the previous one.
    """
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap   # slide forward by (chunk_size - overlap)

    return chunks
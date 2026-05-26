from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from parser import parse_document, chunk_text
from embedder import store_chunks, query_chunks, list_documents

app = FastAPI(
    title="Semantic Document Search API",
    description=(
        "Upload PDF, DOCX, or TXT files and search them in plain English "
        "using sentence-level semantic embeddings."
    ),
    version="1.0.0",
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


# --------------------------------------------------------------------------- #
# Schemas                                                                      #
# --------------------------------------------------------------------------- #

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResult(BaseModel):
    text: str
    source: str
    chunk_index: int
    score: float


# --------------------------------------------------------------------------- #
# Endpoints                                                                    #
# --------------------------------------------------------------------------- #

@app.post(
    "/upload",
    summary="Upload and index a document",
    response_description="Number of chunks stored and source filename",
)
async def upload_document(file: UploadFile = File(...)):
    """
    Accept a **PDF**, **DOCX**, or **TXT** file, parse its text, split it into
    200-word overlapping chunks, embed each chunk with `all-MiniLM-L6-v2`, and
    store the vectors in ChromaDB.
    """
    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        text = parse_document(filename, file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse document: {exc}")

    if not text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the document.")

    chunks = chunk_text(text, chunk_size=200, overlap=50)
    num_stored = store_chunks(filename, chunks)

    return JSONResponse(
        status_code=200,
        content={
            "message": "Document indexed successfully.",
            "filename": filename,
            "chunks_stored": num_stored,
        },
    )


@app.post(
    "/query",
    summary="Search indexed documents",
    response_model=list[QueryResult],
)
async def query_documents(body: QueryRequest):
    """
    Submit a plain-English question and receive the **top-k** most semantically
    relevant passages from all indexed documents, along with the source filename
    and a cosine similarity score (0 – 1, higher = more relevant).
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    if not (1 <= body.top_k <= 20):
        raise HTTPException(status_code=400, detail="`top_k` must be between 1 and 20.")

    results = query_chunks(body.question, top_k=body.top_k)

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No documents have been indexed yet. Upload some documents first.",
        )

    return results


@app.get(
    "/documents",
    summary="List all indexed documents",
    response_description="List of indexed filenames and their chunk counts",
)
async def get_documents():
    """
    Return every unique document that has been indexed, along with the number
    of chunks stored for each one.
    """
    docs = list_documents()
    return {"total_documents": len(docs), "documents": docs}


# --------------------------------------------------------------------------- #
# Entry point (for direct `python main.py` usage)                             #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
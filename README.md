# Semantic Document Search API

A lightweight REST API that lets you **upload documents and search them in plain English**. It chunks your files, embeds every chunk with a sentence-transformer model, and stores the vectors in a local ChromaDB database. At query time it embeds your question the same way and returns the three most semantically similar passages — no keyword matching, no SQL, just meaning.

---

## Stack

| Layer | Library | Role |
|---|---|---|
| API | **FastAPI** | HTTP endpoints, request validation, OpenAPI docs |
| Parsing | **PyMuPDF**, **python-docx** | Extract raw text from PDF / DOCX / TXT |
| Embedding | **Sentence-Transformers** (`all-MiniLM-L6-v2`) | 384-dim dense vectors, fast & accurate |
| Vector store | **ChromaDB** (persistent, local) | Store + cosine-similarity search over embeddings |
| Runtime | **Uvicorn** | ASGI server |

---

## Project Structure

```
semantic-doc-search/
├── main.py          # FastAPI app — endpoint definitions
├── embedder.py      # SentenceTransformer + ChromaDB logic
├── parser.py        # PDF / DOCX / TXT parsing + overlapping chunker
├── requirements.txt
└── README.md
```

A `chroma_store/` directory is created automatically on first run and holds the persistent vector database.

---

## How It Works

```
Upload flow
───────────
File bytes → parse text → split into 200-word chunks (50-word overlap)
          → embed with all-MiniLM-L6-v2 → upsert into ChromaDB

Query flow
──────────
Question string → embed with all-MiniLM-L6-v2
               → cosine similarity search in ChromaDB
               → return top-3 chunks with source + score
```

### Why overlapping chunks?
A 50-word overlap between consecutive chunks ensures that sentences near chunk boundaries are never split in a way that loses context. A relevant passage is far less likely to be missed.

---

## Endpoints

### `POST /upload`
Upload and index a document.

**Form field:** `file` — a `.pdf`, `.docx`, or `.txt` file.

**Response:**
```json
{
  "message": "Document indexed successfully.",
  "filename": "report.pdf",
  "chunks_stored": 42
}
```

---

### `POST /query`
Search indexed documents with a plain-English question.

**Request body:**
```json
{
  "question": "What are the key findings on climate change?",
  "top_k": 3
}
```

`top_k` is optional (default `3`, max `20`).

**Response:**
```json
[
  {
    "text": "The report concludes that global temperatures have risen...",
    "source": "climate_report.pdf",
    "chunk_index": 7,
    "score": 0.891
  },
  ...
]
```

`score` is a cosine similarity value between 0 and 1 — higher means more relevant.

---

### `GET /documents`
List every document that has been indexed.

**Response:**
```json
{
  "total_documents": 2,
  "documents": [
    { "filename": "climate_report.pdf", "chunks": 42 },
    { "filename": "notes.txt",          "chunks": 5  }
  ]
}
```

---

## Setup & Running

### 1. Clone / copy the project

```bash
git clone <your-repo-url>
cd semantic-doc-search
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> The first run will also download the `all-MiniLM-L6-v2` model (~90 MB) from Hugging Face automatically.

### 4. Start the server

```bash
uvicorn main:app --reload
# or simply:
python main.py
```

The API is now live at **http://localhost:8000**.

### 5. Explore the interactive docs

Open **http://localhost:8000/docs** for the Swagger UI where you can upload files and run queries directly in the browser.

---

## Example Usage (curl)

```bash
# Upload a PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/your/document.pdf"

# Query it
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the document say about revenue?", "top_k": 3}'

# List all indexed documents
curl http://localhost:8000/documents
```

---

## Notes & Limitations

- **Re-uploading** the same filename overwrites its chunks (upsert semantics via SHA-256 chunk IDs).
- The vector store persists to `./chroma_store/` — delete that folder to start fresh.
- `all-MiniLM-L6-v2` is CPU-friendly and handles English text best. For multilingual documents, swap in `paraphrase-multilingual-MiniLM-L12-v2`.
- For very large files (100 MB+), consider streaming the upload and processing chunks in batches.
# Study Buddy — Backend

A FastAPI backend implementing a RAG (Retrieval-Augmented Generation) pipeline that answers questions from user-uploaded PDF notes.

Powers the [Study Buddy Flutter app](https://github.com/Msaqib295/study-buddy-app).

## How it works

1. **Upload** — a PDF is uploaded, text is extracted (with OCR fallback for scanned documents via Tesseract), split into overlapping chunks, embedded, and stored in ChromaDB
2. **Ask** — a question is embedded and matched against the most relevant stored chunks; those chunks are passed to an LLM (via Groq) along with the question to generate a grounded answer
3. All data is scoped per-user via Firebase Auth UIDs, so each user's documents and questions are private

## Tech stack

- **FastAPI** — REST API
- **ChromaDB** — vector storage and retrieval
- **sentence-transformers** — text embeddings
- **Groq API** (`openai/gpt-oss-20b`) — answer generation
- **pypdf** + **pytesseract/pdf2image** — PDF text extraction, with OCR fallback for scanned documents

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Upload a PDF, chunk and store it for a given `user_id` |
| POST | `/ask` | Ask a question, get a grounded answer with source documents |
| GET | `/documents` | List a user's uploaded documents |
| DELETE | `/documents/{filename}` | Delete a specific document |
| DELETE | `/user/{user_id}` | Delete all data for a user (used on account deletion) |

## Running locally

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Create a `.env` file:

GROQ_API_KEY=your_key_here


Run:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Deployment

Currently deployed on an Oracle Cloud Always Free tier VM, running as a persistent `systemd` service.

## Notes

- Scanned/image-based PDFs are supported via OCR (requires Tesseract and Poppler installed on the host)
- Embedding generation is CPU-bound; performance scales with document length and available server resources

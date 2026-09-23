from fastapi import FastAPI, UploadFile, File
import chromadb
from groq import Groq
from dotenv import load_dotenv
from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_bytes
import os
import io

load_dotenv(override=True)
client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./study_buddy_db")
collection = chroma_client.get_or_create_collection(name="notes")

app = FastAPI()


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def extract_text_from_pdf_bytes(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    # If no real text was found, this is likely a scanned/image-based PDF
    if not text.strip():
        images = convert_from_bytes(pdf_bytes)
        for image in images:
            text += pytesseract.image_to_string(image) + "\n"

    return text


@app.post("/upload")
async def upload_notes(user_id: str, file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    text = extract_text_from_pdf_bytes(pdf_bytes)

    chunks = chunk_text(text)
    ids = [f"{user_id}_{file.filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"user_id": user_id, "filename": file.filename} for _ in chunks]

    collection.add(documents=chunks, ids=ids, metadatas=metadatas)

    return {"message": f"Added {len(chunks)} chunks from {file.filename}"}


@app.post("/ask")
async def ask(question: str, user_id: str):
    results = collection.query(
        query_texts=[question],
        n_results=5,
        where={"user_id": user_id},
    )
    retrieved_chunks = results['documents'][0]
    retrieved_metadatas = results['metadatas'][0]
    context = "\n\n".join(retrieved_chunks)

    sources = set()
    for metadata in retrieved_metadatas:
        sources.add(metadata['filename'])

    prompt = f"""Answer the question using ONLY the context below. 
If the answer isn't in the context, say "I don't have that information in your notes."

Context:
{context}

Question: {question}

Answer:"""

    response = client_groq.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": list(sources)
    }


@app.get("/documents")
async def list_documents(user_id: str):
    all_items = collection.get(where={"user_id": user_id})
    filenames = set()
    for metadata in all_items['metadatas']:
        filenames.add(metadata['filename'])
    return {"documents": list(filenames)}


@app.delete("/documents/{filename}")
async def delete_document(filename: str, user_id: str):
    all_items = collection.get(where={"user_id": user_id})
    ids_to_delete = [
        chunk_id for chunk_id, metadata in zip(all_items['ids'], all_items['metadatas'])
        if metadata['filename'] == filename
    ]
    if not ids_to_delete:
        return {"message": f"No document found with name {filename}"}

    collection.delete(ids=ids_to_delete)
    return {"message": f"Deleted {len(ids_to_delete)} chunks from {filename}"}

@app.delete("/user/{user_id}")
async def delete_user_data(user_id: str):
    collection.delete(where={"user_id": user_id})
    return {"message": f"All data deleted for user {user_id}"}
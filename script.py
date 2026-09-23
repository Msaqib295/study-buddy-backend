import chromadb
from groq import Groq
from dotenv import load_dotenv
from pypdf import PdfReader
import os

load_dotenv(override=True)
client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./study_buddy_db")
collection = chroma_client.get_or_create_collection(name="notes")


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def add_notes(text, source_name="notes"):
    chunks = chunk_text(text)
    ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    print(f"Added {len(chunks)} chunks from {source_name}")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    return text


def ask_question(question, n_results=5):
    results = collection.query(query_texts=[question], n_results=n_results)
    retrieved_chunks = results['documents'][0]
    context = "\n\n".join(retrieved_chunks)

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

    return response.choices[0].message.content


if __name__ == "__main__":
       pdf_text = extract_text_from_pdf("A.pdf")
       add_notes(pdf_text, source_name="lecture_1") 
       answer = ask_question(input(" Ask any question related to your notes: "))
       print(answer)

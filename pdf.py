import os
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
import streamlit as st

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return embedder.encode(texts)

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def build_index(chunks):
    embeddings = embed_texts(chunks)
    index = NearestNeighbors(n_neighbors=3, metric="euclidean")
    index.fit(embeddings)
    return index, embeddings

def retrieve_chunks(query, chunks, index, embeddings, k=3):
    query_emb = embed_texts([query])
    distances, indices = index.kneighbors(query_emb, return_distance=True, n_neighbors=k)
    return distances[0], [chunks[i] for i in indices[0]]

def ask_question(query, chunks, index, embeddings, threshold=0.5):
    distances, retrieved = retrieve_chunks(query, chunks, index, embeddings)
    if all(d > threshold for d in distances):
        return "I'm sorry, I can only answer questions related to the uploaded PDF."

    context = "\n".join(retrieved)
    prompt = f"""
    You are a helpful assistant. Use only the context below to answer the question.
    If the answer is not in the context, reply: 
    "I'm sorry, I can only answer questions related to the uploaded PDF."

    Context:
    {context}

    Question: {query}
    Answer:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("PDF-based RAG Assistant")
uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")
query = st.text_input("Ask a question about the PDF:")

if uploaded_pdf and query:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_pdf)
        chunks = chunk_text(text)
        index, embeddings = build_index(chunks)
        answer = ask_question(query, chunks, index, embeddings)
    st.success("Answer:")
    st.write(answer)

import os
import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Try importing streamlit (won’t work in debug mode if not installed)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# -----------------------------
# 1️⃣ OpenAI client
# -----------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# 2️⃣ Initialize embedder
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return embedder.encode(texts)

# -----------------------------
# 3️⃣ Build FAISS index from CSVs
# -----------------------------
def build_index_from_csv(files):
    chunks = []
    for file in files:
        df = pd.read_csv(file)
        text_data = df.astype(str).agg(" ".join, axis=1).tolist()
        chunks.extend(text_data)

    embeddings = embed_texts(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))
    return index, chunks

# -----------------------------
# 4️⃣ Retrieve relevant chunks
# -----------------------------
def retrieve_chunks(query, chunks, index, k=3):
    query_emb = embed_texts([query])
    distances, indices = index.search(np.array(query_emb, dtype=np.float32), k)
    return distances[0], [chunks[i] for i in indices[0]]

# -----------------------------
# 5️⃣ Ask question with CSV RAG
# -----------------------------
def ask_question(query, chunks, index, threshold=0.5):
    distances, retrieved = retrieve_chunks(query, chunks, index)

    if all(d > threshold for d in distances):
        return "I'm sorry, I can only answer questions related to the uploaded CSV files."

    context = "\n".join(retrieved)
    prompt = f"""
    You are a helpful assistant. Use only the context below to answer the question.
    If the answer is not in the context, reply: 
    "I'm sorry, I can only answer questions related to the uploaded CSV files."

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

# -----------------------------
# 6️⃣ Streamlit UI
# -----------------------------
def run_streamlit():
    st.title("CSV-based RAG Assistant")
    uploaded_files = st.file_uploader(
        "Upload CSV files", accept_multiple_files=True, type=["csv"]
    )
    user_query = st.text_input("Ask a question about the CSV data:")

    if uploaded_files and user_query:
        with st.spinner("Processing..."):
            index, chunks = build_index_from_csv(uploaded_files)
            answer = ask_question(user_query, chunks, index)
        st.success("Answer:")
        st.write(answer)

# -----------------------------
# 7️⃣ Debug mode (for Jupyter / python app.py)
# -----------------------------
def run_debug():
    print("🔍 Running in debug mode (no Streamlit UI)...")
    sample_file = "your_sample.csv"  # change to one of your CSVs
    query = "What does this CSV contain?"
    index, chunks = build_index_from_csv([sample_file])
    answer = ask_question(query, chunks, index)
    print("Q:", query)
    print("A:", answer)

# -----------------------------
# 8️⃣ Entry point
# -----------------------------
if __name__ == "__main__":
    if STREAMLIT_AVAILABLE and os.getenv("RUN_STREAMLIT", "1") == "1":
        run_streamlit()
    else:
        run_debug()

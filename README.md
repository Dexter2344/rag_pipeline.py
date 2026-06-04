"""
RAG Pipeline on a Phone (v2 - Upgraded)
A local Retrieval-Augmented Generation system running on Android via Termux.
Uses Gemma 4 E4B, Ollama native API, Python, and persistent memory.

Modifications from v1:
- Native Ollama API calls instead of subprocess
- Persistent cross-session memory via chat_memory.json
- Upgraded to Gemma 4 E4B for better reasoning
"""

import os
import json
import requests
import numpy as np
from pathlib import Path

# PDF parsing
try:
    from PyPDF2 import PdfReader
except ImportError:
    print("PyPDF2 not found. Install it: pip install PyPDF2")
    exit(1)

# Vector storage
try:
    import lancedb
except ImportError:
    print("LanceDB not found. Install: pip install lancedb numpy")
    exit(1)

# ============================================================
# CONFIGURATION
# ============================================================
OLLAMA_MODEL = "gemma4:4b"  # Upgraded to E4B
OLLAMA_API = "http://localhost:11434/api/generate"
DB_PATH = "./rag_db"
MEMORY_FILE = "./chat_memory.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ============================================================
# 1. PERSISTENT MEMORY (Cross-Session)
# ============================================================

def load_memory():
    """Load past conversation summaries from JSON file."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {"sessions": [], "last_summary": ""}


def save_memory(memory):
    """Save conversation memory to JSON file."""
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def update_memory(question, answer):
    """Add a new Q&A pair to memory and create a rolling summary."""
    memory = load_memory()
    
    memory["sessions"].append({
        "question": question,
        "answer": answer[:200]  # Store truncated version
    })
    
    # Keep only last 10 interactions
    if len(memory["sessions"]) > 10:
        memory["sessions"] = memory["sessions"][-10:]
    
    # Create a brief rolling summary
    if len(memory["sessions"]) >= 3:
        recent = memory["sessions"][-3:]
        topics = [r["question"][:50] for r in recent]
        memory["last_summary"] = f"Recently discussed: {'; '.join(topics)}"
    
    save_memory(memory)
    return memory


def get_memory_context():
    """Retrieve memory context to inject into prompts."""
    memory = load_memory()
    context = ""
    
    if memory["last_summary"]:
        context += f"## Previous Conversation Summary\n{memory['last_summary']}\n\n"
    
    recent_qa = memory["sessions"][-3:] if memory["sessions"] else []
    if recent_qa:
        context += "## Recent Q&A\n"
        for qa in recent_qa:
            context += f"Q: {qa['question']}\nA: {qa['answer']}...\n\n"
    
    return context


# ============================================================
# 2. SIMPLE LOCAL EMBEDDING (Phone-Friendly)
# ============================================================

def simple_embed(text, dim=128):
    """Lightweight character n-gram hashing embedding."""
    text = text.lower()
    vec = np.zeros(dim)
    
    for i in range(len(text) - 1):
        bigram = text[i:i+2]
        idx = hash(bigram) % dim
        vec[idx] += 1
    
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    
    return vec


# ============================================================
# 3. PDF PARSING & CHUNKING
# ============================================================

def parse_pdf(pdf_path):
    """Extract text from a PDF file."""
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return None
    
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    
    return text.strip()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    
    return chunks


# ============================================================
# 4. VECTOR DATABASE
# ============================================================

def setup_database():
    """Create or connect to LanceDB."""
    db = lancedb.connect(DB_PATH)
    return db


def store_chunks(chunks):
    """Embed and store text chunks."""
    db = setup_database()
    
    records = []
    for i, chunk in enumerate(chunks):
        vec = simple_embed(chunk)
        records.append({
            "id": i,
            "text": chunk,
            "vector": vec.tolist()
        })
    
    table = db.create_table("documents", data=records, mode="overwrite")
    print(f"Stored {len(chunks)} chunks in the database.")
    return table


def search_similar(query, top_k=3):
    """Find the most relevant chunks for a query."""
    db = setup_database()
    table = db.open_table("documents")
    
    query_vec = simple_embed(query)
    df = table.to_pandas()
    
    similarities = []
    for i, row in df.iterrows():
        chunk_vec = np.array(row["vector"])
        sim = np.dot(query_vec, chunk_vec)
        similarities.append((sim, row["text"]))
    
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in similarities[:top_k]]


# ============================================================
# 5. NATIVE OLLAMA API (Replaces Subprocess)
# ============================================================

def ask_ollama(prompt):
    """
    Send prompt to Gemma 4 E4B via Ollama's native REST API.
    This is cleaner and faster than subprocess.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 512
        }
    }
    
    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json().get("response", "No response from model.")
        else:
            return f"API error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "Cannot connect to Ollama. Make sure it's running in Termux."
    except requests.exceptions.Timeout:
        return "Model took too long. Try a shorter query."


# ============================================================
# 6. FULL RAG PIPELINE WITH MEMORY
# ============================================================

def ask_question(question, top_k=3):
    """
    Full upgraded RAG pipeline:
    1. Retrieve relevant document chunks
    2. Load conversation memory
    3. Build prompt with context + memory
    4. Ask Gemma 4 E4B via native API
    5. Save interaction to memory
    """
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    # Step 1: Retrieve relevant chunks
    relevant_chunks = search_similar(question, top_k=top_k)
    doc_context = "\n\n".join(relevant_chunks)
    
    # Step 2: Load memory context
    memory_context = get_memory_context()
    
    # Step 3: Build full prompt with document context + memory
    prompt = f"""You are a helpful AI assistant running locally on a phone. Answer the question based on the document context provided. Use the conversation history for continuity.

## Document Context
{doc_context}

## Conversation History
{memory_context if memory_context else "No previous conversation."}

## Current Question
{question}

ANSWER:"""
    
    # Step 4: Ask the local LLM via native API
    response = ask_ollama(prompt)
    
    # Step 5: Save to persistent memory
    update_memory(question, response)
    
    return response


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def prepare_document(pdf_path):
    """Full ingestion pipeline."""
    print(f"Processing: {pdf_path}")
    
    text = parse_pdf(pdf_path)
    if not text:
        print("Failed to extract text from PDF.")
        return False
    
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks.")
    
    store_chunks(chunks)
    print("Document ready for questions.")
    return True


def interactive_mode():
    """Interactive Q&A loop with persistent memory."""
    print("\n" + "=" * 50)
    print("RAG PIPELINE v2 - Upgraded")
    print(f"Model: {OLLAMA_MODEL}")
    print("Features: Native API + Persistent Memory")
    print("Type 'exit' to quit, 'memory' to see history.")
    print("=" * 50)
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ["exit", "quit", "q"]:
            print("Shutting down. Goodbye!")
            break
        
        if question.lower() == "memory":
            mem = load_memory()
            print("\n--- Conversation History ---")
            if mem["sessions"]:
                for qa in mem["sessions"]:
                    print(f"Q: {qa['question']}")
                    print(f"A: {qa['answer']}...\n")
            else:
                print("No history yet.")
            continue
        
        if not question:
            continue
        
        answer = ask_question(question)
        print(f"\nAnswer: {answer}")


# ============================================================
# 8. RUN
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python rag_pipeline.py prepare <pdf_path>")
        print("  python rag_pipeline.py ask")
        print("\nExample:")
        print("  python rag_pipeline.py prepare my_notes.pdf")
        print("  python rag_pipeline.py ask")
    else:
        command = sys.argv[1]
        
        if command == "prepare":
            if len(sys.argv) < 3:
                print("Please provide a PDF path.")
            else:
                prepare_document(sys.argv[2])
        
        elif command == "ask":
            interactive_mode()
        
        else:
            print(f"Unknown command: {command}")

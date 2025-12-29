# seed_chroma.py
"""
Seeds local ChromaDB with medical QnA and medical device documents.
Run ONCE before starting Streamlit or agent.
"""

import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# ----------------------------
# Config
# ----------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "processed"
#CHROMA_PATH = BASE_DIR / "chroma_db"
# Use persistent path if provided (Railway), else local
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", BASE_DIR / "chroma_db"))

QNA_FILE = DATA_DIR / "medical_qna.txt"
DEVICE_FILE = DATA_DIR / "medical_devices.txt"

QNA_COLLECTION = "medical_qna_collection"
DEVICE_COLLECTION = "medical_devices_collection"

# --------------------------------------------------
# Embedding model
# --------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 500

# ----------------------------
# Load embedding model
# ----------------------------
print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ----------------------------
# Initialize Chroma
# ----------------------------
print("Initializing ChromaDB...")
client = chromadb.PersistentClient(path=str(CHROMA_PATH))

qna_collection = client.get_or_create_collection(name=QNA_COLLECTION)
device_collection = client.get_or_create_collection(name=DEVICE_COLLECTION)

# ----------------------------
# Helper: load text chunks
# ----------------------------
def load_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    chunks = [c.strip() for c in content.split("----") if c.strip()]
    return chunks

# ----------------------------
# Seed function
# ----------------------------
#def seed_collection(collection, chunks, source_name):
#    print(f"Seeding {source_name} ({len(chunks)} chunks)...")

#    embeddings = embedder.encode(chunks, show_progress_bar=True)

#    ids = [f"{source_name}_{i}" for i in range(len(chunks))]
#    metadatas = [{"source": source_name} for _ in chunks]
#    print(f"Adding in collection...")
#    collection.add(
#        documents=chunks,
#        embeddings=embeddings.tolist(),
#        ids=ids,
#        metadatas=metadatas
#    )

#    print(f"✅ {source_name} seeded successfully.")

def seed_collection(collection, chunks, source_name, batch_size=500):
    print(f"Seeding {source_name} ({len(chunks)} chunks)...")

    total = len(chunks)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_chunks = chunks[start:end]

        print(f"Embedding chunks {start + 1} → {end}...")
        embeddings = embedder.encode(
            batch_chunks,
            show_progress_bar=False
        )

        ids = [f"{source_name}_{i}" for i in range(start, end)]
        metadatas = [{"source": source_name} for _ in batch_chunks]

        print(f"Adding batch {start + 1} → {end} to collection...")
        collection.add(
            documents=batch_chunks,
            embeddings=embeddings.tolist(),
            ids=ids,
            metadatas=metadatas
        )

    print(f"✅ {source_name} seeded successfully.")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    if not QNA_FILE.exists():
        raise FileNotFoundError(f"Missing file: {QNA_FILE}")

    if not DEVICE_FILE.exists():
        raise FileNotFoundError(f"Missing file: {DEVICE_FILE}")

    qna_chunks = load_chunks(QNA_FILE)
    device_chunks = load_chunks(DEVICE_FILE)

    seed_collection(qna_collection, qna_chunks, "medical_qna",BATCH_SIZE)
    seed_collection(device_collection, device_chunks, "medical_devices")

    print("\n🎉 Chroma DB seeding complete.")
    print(f"📂 DB location: {CHROMA_PATH}")

import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from src.data_loader import load_or_create_medical_df

# Initialize local persistent ChromaDB and SentenceTransformer
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

class SimpleTextSplitter:
    """Pure Python text splitter with overlap (zero external dependency issues)."""
    def __init__(self, chunk_size=650, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += (self.chunk_size - self.chunk_overlap)
        return chunks

med_splitter = SimpleTextSplitter(chunk_size=650, chunk_overlap=100)

def get_collection():
    return chroma_client.get_or_create_collection(
        name="medical_rag_kb",
        metadata={"hnsw:space": "cosine"}
    )

def index_medical_data(extra_files=None) -> str:
    df_med = load_or_create_medical_df()
    docs, metadatas, ids = [], [], []

    # 1. Index Structured Clinical FAQ Dataset with Context Injection
    for _, row in df_med.iterrows():
        enriched_chunk = (
            f"CONDITION: {row['condition']} ({row['category']})\n"
            f"PATIENT QUESTION: {row['question']}\n"
            f"CLINICAL GUIDELINE ANSWER: {row['answer']}"
        )
        docs.append(enriched_chunk)
        metadatas.append({
            "id": row["id"],
            "category": row["category"],
            "condition": row["condition"],
            "source": row["source"]
        })
        ids.append(row["id"])

    # 2. Optional: Index custom uploaded Medical PDFs
    if extra_files:
        chunk_idx = 1
        for file_path in extra_files:
            fname = os.path.basename(file_path)
            if fname.lower().endswith(".pdf"):
                reader = PdfReader(file_path)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if not text or not text.strip():
                        continue
                    for ch in med_splitter.split_text(" ".join(text.split())):
                        docs.append(f"UPLOADED DOCUMENT ({fname}, Page {page_num+1}):\n{ch}")
                        metadatas.append({
                            "id": f"PDF-{chunk_idx}",
                            "category": "Uploaded Medical PDF",
                            "condition": fname,
                            "source": f"{fname} (Page {page_num+1})"
                        })
                        ids.append(f"PDF-{chunk_idx}")
                        chunk_idx += 1

    # Reset collection cleanly and insert embeddings
    try:
        chroma_client.delete_collection(name="medical_rag_kb")
    except Exception:
        pass

    med_collection = chroma_client.create_collection(
        name="medical_rag_kb",
        metadata={"hnsw:space": "cosine"}
    )

    embeddings = embed_model.encode(docs).tolist()
    med_collection.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
    return f"✅ Indexed {len(docs)} clinical knowledge chunks into ChromaDB!"
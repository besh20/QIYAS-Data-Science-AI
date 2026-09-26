"""RAG over the knowledge base. Chunks by markdown section, embeds with a
free multilingual model, retrieves top-k with FAISS, and asks Gemini to
answer using only the retrieved context.
"""
import os
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

KB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.md")

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("intfloat/multilingual-e5-small")
    return _model


def load_chunks():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    # split on markdown ## headers
    parts = re.split(r"\n(?=## )", text)
    return [p.strip() for p in parts if p.strip()]


class RagIndex:
    def __init__(self):
        self.chunks = load_chunks()
        model = get_model()
        # e5 models expect "passage: " / "query: " prefixes
        embeddings = model.encode([f"passage: {c}" for c in self.chunks], normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.array(embeddings, dtype="float32"))

    def retrieve(self, query: str, k: int = 3):
        model = get_model()
        q_emb = model.encode([f"query: {query}"], normalize_embeddings=True)
        scores, idxs = self.index.search(np.array(q_emb, dtype="float32"), k)
        return [(self.chunks[i], float(scores[0][j])) for j, i in enumerate(idxs[0]) if i != -1]


if __name__ == "__main__":
    idx = RagIndex()
    for chunk, score in idx.retrieve("how much is registration"):
        print(round(score, 3), chunk[:60])

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class LocalVectorDB:
    def __init__(self, storage_dir: str | Path, model_name: str = "all-MiniLM-L6-v2"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()

        self.index_path = self.storage_dir / "index.faiss"
        self.metadata_path = self.storage_dir / "metadata.json"

        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexFlatIP(self.dimension)

        self.metadata: List[Dict[str, Any]] = []
        if self.metadata_path.exists():
            try:
                self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.metadata = []

    def _embed(self, text: str) -> np.ndarray:
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embedding, dtype="float32")

    def add_documents(self, chunks: List[Dict[str, str]]) -> int:
        if not chunks:
            return 0

        texts = [chunk["text"] for chunk in chunks]
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        if self.index.d != vectors.shape[1]:
            self.index = faiss.IndexFlatIP(vectors.shape[1])

        self.index.add(vectors)
        self.metadata.extend({"text": chunk["text"], "source": chunk["source"]} for chunk in chunks)
        self._save()
        return len(chunks)

    def search(self, query: str, top_k: int = 3, min_score: float = 0.0) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        query_vector = self._embed(query)
        scores, indexes = self.index.search(query_vector, min(top_k, self.index.ntotal))

        results: List[Dict[str, Any]] = []
        for score, index in zip(scores[0], indexes[0]):
            if index < 0:
                continue
            if float(score) < min_score:
                continue

            record = self.metadata[int(index)]
            results.append({
                "text": record["text"],
                "source": record["source"],
                "score": float(score),
            })

        return results

    def reset(self) -> None:
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []

    def count(self) -> int:
        return self.index.ntotal

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

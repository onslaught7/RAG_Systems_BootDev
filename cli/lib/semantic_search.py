import os
import logging

# ---- Silence HuggingFace / Transformers logs ----
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer
import numpy as np
import json
import re


def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def fixed_size_chunking(text: str, chunk_length: int = 200, overlap: int = 0) -> list:
    words = text.split()
    step = chunk_length - overlap
    chunks = []

    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_length]
        chunks.append(" ".join(chunk))

    return chunks


def semantic_chunk_text(text: str, chunk_size: int = 4, overlap: int = 0) -> list:
    sentences = _split_into_sentences(text)

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(sentences), step):
        chunk_content = sentences[i:i + chunk_size]
        chunks.append(" ".join(chunk_content))

        if i + chunk_size >= len(sentences):
            break

    return chunks


def _split_into_sentences(text: str) -> list:
    if not text.strip():
        return []

    pattern = r"(?<=[.!?])\s+"
    sentences = re.split(pattern, text.strip())

    if len(sentences) == 1 and not text.strip().endswith((".", "!", "?")):
        return [text.strip()]

    return [s for s in sentences if s.strip()]


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def build_embeddings(self, documents):
        self.documents = documents
        doc_string_list = []

        for document in documents:
            self.document_map[document["id"]] = document
            doc_string = f"{document['title']}: {document['description']}"
            doc_string_list.append(doc_string)

        self.embeddings = self.model.encode(
            doc_string_list,
            show_progress_bar=False
        )

        os.makedirs("cache", exist_ok=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        doc_string_list = []

        for document in documents:
            self.document_map[document["id"]] = document
            doc_string = f"{document['title']}: {document['description']}"
            doc_string_list.append(doc_string)

        cache_path = "cache/movie_embeddings.npy"

        if os.path.exists(cache_path):
            self.embeddings = np.load(cache_path)

            if len(self.embeddings) == len(documents):
                return self.embeddings

        self.build_embeddings(documents)
        return self.embeddings

    def generate_embedding(self, text) -> np.ndarray:
        if not text.strip():
            raise ValueError("Text cannot be empty or whitespace.")

        text_embedded = self.model.encode([text], show_progress_bar=False)
        return text_embedded[0]

    def search(self, query, limit) -> list:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call load_or_create_embeddings first.")

        query_embedded = self.generate_embedding(query)

        similarity_list = []

        for i in range(len(self.embeddings)):
            similarity = cosine_similarity(query_embedded, self.embeddings[i])
            similarity_list.append((similarity, self.documents[i]))

        similarity_list.sort(key=lambda x: x[0], reverse=True)

        results = []

        for score, doc in similarity_list[:limit]:
            results.append({
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            })

        return results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.chunk_metadata = []
        all_chunks = []

        for i, document in enumerate(documents):
            if not document["description"]:
                continue

            self.document_map[document["id"]] = document

            chunked_description = semantic_chunk_text(document["description"], 4, 1)

            for j, chunk in enumerate(chunked_description):
                all_chunks.append(chunk)
                self.chunk_metadata.append({
                    "movie_idx": i,
                    "chunk_idx": j,
                    "total_chunks": len(chunked_description)
                })

        self.chunk_embeddings = self.model.encode(
            all_chunks,
            show_progress_bar=False
        )

        os.makedirs("cache", exist_ok=True)
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)

        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({
                "chunks": self.chunk_metadata,
                "total_chunks": len(all_chunks)
            }, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        for document in documents:
            self.document_map[document["id"]] = document

        embedding_path = "cache/chunk_embeddings.npy"
        metadata_path = "cache/chunk_metadata.json"

        if os.path.exists(embedding_path) and os.path.exists(metadata_path):
            self.chunk_embeddings = np.load(embedding_path)

            with open(metadata_path, "r") as f:
                metadata_json = json.load(f)
                self.chunk_metadata = metadata_json["chunks"]

            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list:
        embedded_query = self.generate_embedding(query)

        movie_score_dict = {}

        for i in range(len(self.chunk_embeddings)):
            similarity = cosine_similarity(embedded_query, self.chunk_embeddings[i])
            metadata = self.chunk_metadata[i]
            movie_idx = metadata["movie_idx"]

            if movie_idx not in movie_score_dict or similarity > movie_score_dict[movie_idx]:
                movie_score_dict[movie_idx] = similarity

        sorted_movies = sorted(
            movie_score_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for movie_idx, score in sorted_movies[:limit]:
            doc = self.documents[movie_idx]

            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"][:100],
                "score": round(score, 4),
                "metadata": {}
            })

        return results
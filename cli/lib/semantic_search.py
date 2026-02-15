from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import re


def verify_model():
    smn_search = SemanticSearch()
    model = smn_search.model

    print(f"Model Loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")


def verify_embeddings():
    smn_search = SemanticSearch()
    
    with open("data/movies.json", "r") as f:
        data = json.load(f)
        documents = data["movies"]
        embeddings = smn_search.load_or_create_embeddings(documents)

        print(f"Number of docs:   {len(documents)}")
        print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_query_text(query):
    smn_search = SemanticSearch()
    embedding = smn_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
    

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


def chunk_text(text: str, chunk_size: int, overlap: int):
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")    


def semantic_chunk_text(text: str, chunk_size: int = 4, overlap: int = 0) -> list:
    sentences = _split_into_sentences(text)
    
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(sentences), step):
        chunk_content = sentences[i:i + chunk_size]

        chunks.append(chunk_content)

        if i + chunk_size >= len(sentences):
            break

    # print(f"Semantically chunking {len(text)} characters")
    # for i, chunk in enumerate(chunks):
    #     chunk_text = " ".join(chunk)
    #     print(f"{i + 1}. {chunk_text}")

    return [" ".join(chunk) for chunk in chunks]


def _split_into_sentences(text: str) -> list:
    pattern = r"(?<=[.!?])\s+"

    sentences = re.split(pattern, text)

    return [s for s in sentences if s.strip()]  


class SemanticSearch:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}


    def build_embeddings(self, documents):
        self.documents = documents
        doc_string_list = []

        for document in documents:
            self.document_map[document['id']] = document
            doc_string = f"{document['title']}: {document['description']}"
            doc_string_list.append(doc_string)

        self.embeddings = self.model.encode(doc_string_list, show_progress_bar=True)
        
        os.makedirs("cache", exist_ok=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)


    def load_or_create_embeddings(self, documents):
        self.documents = documents
        doc_string_list = []

        for document in documents:
            self.document_map[document['id']] = document
            doc_string = f"{document['title']}: {document['description']}"
            doc_string_list.append(doc_string)

        cache_path = "cache/movie_embeddings.npy"

        if os.path.exists(cache_path):
            self.embeddings = np.load(cache_path)

            if len(self.embeddings) == len(documents):
                return self.embeddings
        else:
            self.build_embeddings(documents)
            return self.embeddings

    
    def generate_embedding(self, text) -> str:
        if text == "" or text == " ":
            raise ValueError("Text cannot be empty or a whitespace.")
        
        text_embedded = self.model.encode([text])
        return text_embedded[0]
    

    def search(self, query, limit) -> list:
        embedding_path = "cache/movie_embeddings.npy"

        if os.path.exists(embedding_path) == False:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
    
        query_embedded = self.generate_embedding(query)
        cosine_similarity_lsit = []
        for i in range(0, len(self.embeddings)):
            similarity = cosine_similarity(query_embedded, self.embeddings[i])
            cosine_similarity_lsit.append((similarity, self.documents[i]))

        cosine_similarity_lsit.sort(key=lambda x : x[0], reverse=True)
        
        search_list = []
        for i in range(0, limit):
            score, doc = cosine_similarity_lsit[i]
            search_list.append({
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            })

        return search_list


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> list:
        self.documents = documents
        self.chunk_metadata = []
        all_chunks = []

        for i, document in enumerate(documents):
            if document["description"] == "":
                continue

            self.document_map[document["id"]] = document

            chunked_description = semantic_chunk_text(document["description"], 4, 1)
            for j, chunk in enumerate(chunked_description):
                all_chunks.append(chunk)
                chunk_metadata = {
                    "movie_idx": i,
                    "chunk_idx": j,
                    "total_chunks": len(chunked_description)
                }
                self.chunk_metadata.append(chunk_metadata)

        self.chunk_embeddings = self.model.encode(all_chunks)

        os.makedirs("cache", exist_ok=True)
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)   

        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({
                    "chunks": self.chunk_metadata,
                    "total_chunks": len(all_chunks)
                },
                f,
                indent=2                
            )

        return self.chunk_embeddings


    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document

        embedding_path = "cache/chunk_embeddings.npy"
        metadata_path = "cache/chunk_metadata.json"

        if os.path.exists("cache/chunk_metadata.json") and os.path.exists("cache/chunk_embeddings.npy"):
            self.chunk_embeddings = np.load(embedding_path)

            with open(metadata_path, "r") as f:
                metadata_json = json.load(f)
                self.chunk_metadata = metadata_json["chunks"]

            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)


    def search_chunks(self, query: str, limit: int = 10) -> list:
        embedded_query = self.generate_embedding(query)
        chunk_score = []

        for i in range(len(self.chunk_embeddings)):
            similarity = cosine_similarity(embedded_query, self.chunk_embeddings[i])
            metadata = self.chunk_metadata[i]

            chunk_score.append({
                "chunk_idx": metadata["chunk_idx"],
                "movie_idx": metadata["movie_idx"],
                "score": similarity
            })
        
        movie_score_dict = {}

        for chunk in chunk_score:
            movie_idx = chunk["movie_idx"]
            score = chunk["score"]

            if movie_idx not in movie_score_dict or score > movie_score_dict[movie_idx]:
                movie_score_dict["movie_idx"] = chunk["score"]

        sorted_movies = sorted(
            movie_score_dict.items(), 
            key=lambda x : x[1], 
            reverse=True
        )

        top_movies = sorted_movies[:limit]

        results = []

        for movie_idx, score in top_movies:

            doc = self.documents[movie_idx]

            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"][:100],
                "score": round(score, 4),
                "metadata": {}
            })

        return results
            

def embed_text(text):
    smt_search = SemanticSearch()
    embed_text = smt_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embed_text[:3]}")
    print(f"Dimensions: {embed_text.shape[0]}")
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os


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
    chunks = []
    for i in range(0, len(words), chunk_length):
        if i == 0:
            chunk_content = " ".join(words[i:chunk_length + i])
        else:
            chunk_content = " ".join(words[i - overlap: chunk_length + i - overlap])
        chunks.append(chunk_content)

    return chunks


def chunk_text(text: str, chunk_size: int, overlap: int):
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")    


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
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


def embed_text(text):
    smt_search = SemanticSearch()
    embed_text = smt_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embed_text[:3]}")
    print(f"Dimensions: {embed_text.shape[0]}")
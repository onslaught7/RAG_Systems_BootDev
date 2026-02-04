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
    

def embed_text(text):
    smt_search = SemanticSearch()
    embed_text = smt_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embed_text[:3]}")
    print(f"Dimensions: {embed_text.shape[0]}")
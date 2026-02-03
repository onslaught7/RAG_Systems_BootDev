from sentence_transformers import SentenceTransformer

def verify_model():
    smn_search = SemanticSearch()
    model = smn_search.model

    print(f"Model Loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    

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
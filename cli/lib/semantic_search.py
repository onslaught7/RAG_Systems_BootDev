from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')


def verify_model():
    smn_search = SemanticSearch()
    model = smn_search.model

    print(f"Model Loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")
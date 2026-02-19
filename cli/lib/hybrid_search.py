import os

# local import from the same package
from .semantic_search import ChunkedSemanticSearch
from src.inverted_index import InvertedIndex


def min_max_normalization(scores: list[float]) -> None:
    if len(scores) == 0:
        return

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        for _ in range(len(scores)):
            print(f"* {1.0}")
        return

    normalized_list = []

    for score in scores:
        normalized_score = (score - min_score) / (max_score - min_score)
        normalized_list.append(normalized_score)

    for score in normalized_list:
        print(f"* {score:.4f}")


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists("./cache"):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
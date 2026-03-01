import os

from .semantic_search import ChunkedSemanticSearch
from .keyword_search import InvertedIndex


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
    
    def _semantic_search(self, query, limit):
        return self.semantic_search.search_chunks(query, limit)

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    
    def min_max_normalization(self, scores: list[float]) -> list[float]:
        if len(scores) == 0:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0 for _ in scores]

        normalized_list = []

        for score in scores:
            normalized_score = (score - min_score) / (max_score - min_score)
            normalized_list.append(normalized_score)

        return normalized_list
    
    def weighted_search(self, query: str, alpha: float, limit: int = 5):
        bm25_results = self._bm25_search(query, limit * 5)
        semantic_results = self._semantic_search(query, limit * 5)

        bm25_scores = [score for _, score in bm25_results]
        semantic_scores = [item["score"] for item in semantic_results]

        bm25_norm = self.min_max_normalization(bm25_scores) or []
        semantic_norm = self.min_max_normalization(semantic_scores) or []

        doc_dict = {}

        # Insert BM25
        for i, (doc_id, _) in enumerate(bm25_results):
            doc_dict[doc_id] = {
                "bm25_score": bm25_norm[i],
                "semantic_score": 0.0
            }

        # Insert Semantic
        for i, item in enumerate(semantic_results):
            doc_id = item["id"]

            if doc_id not in doc_dict:
                doc_dict[doc_id] = {
                    "bm25_score": 0.0,
                    "semantic_score": semantic_norm[i]
                }
            else:
                doc_dict[doc_id]["semantic_score"] = semantic_norm[i]

        # Hybrid score
        for doc_id in doc_dict:
            doc_dict[doc_id]["hybrid_score"] = self.hybrid_score(
                doc_dict[doc_id]["bm25_score"],
                doc_dict[doc_id]["semantic_score"],
                alpha
            )

        # Sort
        sorted_docs = sorted(
            doc_dict.items(),
            key=lambda x: x[1]["hybrid_score"],
            reverse=True
        )

        results = []

        for doc_id, scores in sorted_docs[:limit]:
            doc = self.idx.docmap.get(doc_id, {})

            results.append({
                "title": doc.get("title", ""),
                "document": doc.get("description", "")[:150],
                "bm25_score": scores["bm25_score"],
                "semantic_score": scores["semantic_score"],
                "hybrid_score": scores["hybrid_score"]
            })

        return results
    
    def hybrid_score(self, bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
        return alpha * bm25_score + (1 - alpha) * semantic_score
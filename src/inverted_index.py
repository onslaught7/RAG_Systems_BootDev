from typing import Dict, Set, List
from src.text_processing import _normalize_text
from collections import Counter
import json
import pickle
import os


class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, Dict] = {}
        self.term_frequencies: Dict[int, Counter] = {}


    def _add_document(self, doc_id: int, text: str) -> None:
        """Tokenize the input text and add each token to the index with the document ID."""
        tokens = _normalize_text(text)
        token_counts = Counter(tokens)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id] = token_counts


    def get_documents(self, term: str) -> List[int]:
        """Get the set of document IDs for a given token and return them as a list sorted in asc order."""
        normalized = _normalize_text(term)
        if not normalized:
            return []
        token = normalized[0]
        document_ids = self.index.get(token, set())
        
        return sorted(list(document_ids))


    def build(self) -> None:
        """Iterate through all the movies and add them to both the index and the docmap."""
        try:
            with open("./data/movies.json", "r") as f:
                movies_dict = json.load(f)
            for movie in movies_dict.get("movies", []):
                movie_id = movie["id"]
                movie_title = movie.get("title", "")
                movie_desc = movie.get("description", "")
                text = f"{movie_title} {movie_desc}"
                self._add_document(movie_id, text)
                self.docmap[movie_id] = movie
            self.save()
        except FileNotFoundError:
            print("Error: ../data/movies.json not found.")
        except Exception as e:
            print(f"An error occurred while building the index: {e}")


    def save(self) -> None:
        """Save the index and the docmap attributes to the disk."""
        os.makedirs("./cache", exist_ok=True)
        try:
            with open("./cache/index.pkl", "wb") as f:
                pickle.dump(self.index, f)
            with open("./cache/docmap.pkl", "wb") as f:
                pickle.dump(self.docmap, f)
            with open("./cache/term_frequencies.pkl", "wb") as f:
                pickle.dump(self.term_frequencies, f)
        except Exception as e:
            print(f"An error occurred while saving the index: {e}")


    def load(self) -> None:
        """Load the index and the docmap attributes from the disk."""
        try:
            with open("./cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)
            with open("./cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)
            with open("./cache/term_frequencies.pkl", "rb") as f:
                self.term_frequencies = pickle.load(f)
        except FileNotFoundError:
            print("Error: Cache files not found. Please build the index first.")
        except Exception as e:  
            print(f"An error occurred while loading the index: {e}")


    def get_tf(self, doc_id: str, term: str) -> int:
        """Return the times the token appears in the document with the given id"""
        try:
            tokenized_term = _normalize_text(term)

            if len(tokenized_term) == 0:
                return 0
            if len(tokenized_term) > 1:
                raise ValueError("Term must be a single token")

            if term not in self.term_frequencies[doc_id]:
                return 0

            return self.term_frequencies[doc_id][term]
        except Exception as e:
            print(f"An error occurred while getting the term frequency: {e}")
            return 0
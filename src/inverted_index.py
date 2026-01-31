import math
from sys import float_info
from typing import Dict, Set, List
from src.text_processing import _normalize_text
from collections import Counter
from constants.constants import BM25_K1, BM25_B
import json
import pickle
import os


class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, Set[int]] = {}
        self.docmap: Dict[int, Dict] = {}
        self.term_frequencies: Dict[int, Counter] = {}
        self.doc_lengths: Dict[int, int] = {}


    def _add_document(self, doc_id: int, text: str) -> None:
        """Tokenize the input text and add each token to the index with the document ID."""
        tokens = _normalize_text(text)
        token_counts = Counter(tokens)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id] = token_counts
        self.doc_lengths[doc_id] = len(tokens)


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
            with open("./cache/doc_lengths.pkl", "wb") as f:
                pickle.dump(self.doc_lengths, f)
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
            with open("./cache/doc_lengths.pkl", "rb") as f:
                self.doc_lengths = pickle.load(f)
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

            token = tokenized_term[0]
            if token not in self.term_frequencies[doc_id]:
                return 0

            return self.term_frequencies[doc_id][token]
        except Exception as e:
            print(f"An error occurred while getting the term frequency: {e}")
            return 0

    
    def get_idf(self, term: str) -> float:
        """Return the Inverted Term Frequency of the term"""
        try:
            tokenized_term = _normalize_text(term)
            
            if len(tokenized_term) == 0:
                return 0
            if len(tokenized_term) > 1:
                raise ValueError("Term must be a single token")

            token = tokenized_term[0]

            total_docs = len(self.docmap)
            doc_id_set = self.index.get(token, set())
            docs_containing_term = len(doc_id_set)
            
            return math.log((total_docs + 1) / (docs_containing_term + 1))
        except Exception as e:
            print(f"An error occurred while getting the Inverse Document Frequency: {e}")
            return 0


    def get_tfidf(self, doc_id: int, term: str) -> float:
        """Return the TF-IDF score of the term in the doc_id"""
        try:
            tf = self.get_tf(doc_id, term)
            idf = self.get_idf(term)
            return tf * idf
        except Exception as e:
            print(f"An error occurred while calculating TF-IDF: {e}")
            return 0
    
    def get_bm25_idf(self, term: str) -> float:
        """Return the Inverse Document Frrquency of the term using BM25 algorithm"""
        try:
            tokenized_term = _normalize_text(term)

            if len(tokenized_term) == 0:
                    return 0
            if len(tokenized_term) > 1:
                raise ValueError("Term must be a single token") 
            
            token = tokenized_term[0]

            total_docs = len(self.docmap)
            doc_id_set = self.index.get(token, set())
            docs_containing_term = len(doc_id_set)
            docs_not_containing_term = total_docs - docs_containing_term

            return math.log((docs_not_containing_term + 0.5) / (docs_containing_term + 0.5) + 1)
        except Exception as e:
            print(f"An error occurred while getting the BM25 Inverse Document Frequency: {e}")
            return 0
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        """Return the BM25 term frequency for a given term in a document."""
        try:
            tf = self.get_tf(doc_id, term)
            avg_dl = self._get_avg_doc_length()
        
            if avg_dl == 0:
                return 0.0
            
            current_doc_length = self.doc_lengths.get(doc_id, 0)
            length_norm = 1 - b + b * (current_doc_length / avg_dl)
            
            return (tf * (k1 + 1)) / (tf + k1 * length_norm)
        except Exception as e:
            print(f"An error occurred while getting the BM25 term frequency: {e}")
            return 0
        

    def _get_avg_doc_length(self) -> float:
        """Calculate and return the average length accross all documents."""
        try:
            total_docs = len(self.doc_lengths)
            if total_docs == 0:
                return 0.0
            
            total_length = sum(self.doc_lengths.values())

            return total_length / total_docs
                
        except Exception as e:
            print(f"Error getting average doc length")
            return 0.0
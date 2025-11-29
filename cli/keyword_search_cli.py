import sys
from pathlib import Path

from nltk import HeldoutProbDist
# Add parent directory to path so we can import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.text_processing import _normalize_text
from src.inverted_index import InvertedIndex
import argparse # To parse command-line arguments
from typing import List
import json


def search_movies(query: str) -> List[str]:
    """Search movies by title containing the query string."""
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Error: Inverted index not found. Please build the index first using the 'build' command.")
        return []

    normalized_query = _normalize_text(query)
    if not normalized_query:
        return []
    
    matching_doc_ids = set()    
    for token in normalized_query[:5]:
        doc_ids = idx.get_documents(token)
        matching_doc_ids.update(doc_ids)

    results = []
    for doc_id in sorted(matching_doc_ids):
        if doc_id in idx.docmap:
            results.append(idx.docmap[doc_id]["title"])

    return  results[:5]


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get the term frequency")
    tf_parser.add_argument("doc_id", type=int, help="The document id")
    tf_parser.add_argument("term", type=str, help="The term whose frequency is to be found")

    idf_parser = subparsers.add_parser("idf", help="Get the inverse document frequency")
    idf_parser.add_argument("term", type=str, help="The term whose inverse document frequency is to be found")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get the TF-IDF score.")
    tfidf_parser.add_argument("doc_id", type=int, help="The document id")
    tfidf_parser.add_argument("term", type=str, help="The term whose TF-IDF is to be found")

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            results = search_movies(args.query)
            if results:
                for i, result in enumerate(results, start=1):
                    print(f"{i}. {result}")
            else:
                print("No results found.")
        case "build":
            idx = InvertedIndex()
            idx.build()
        case "tf":
            idx = InvertedIndex()
            idx.load()
            tf = idx.get_tf(args.doc_id, args.term)
            if tf > 0:
                print(tf)
            else: 
                print(0)
        case "idf":
            idx = InvertedIndex()
            idx.load()
            idf = idx.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            idx = InvertedIndex()
            idx.load()
            tf_idf = idx.get_tfidf(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
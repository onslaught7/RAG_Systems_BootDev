import sys
from pathlib import Path
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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
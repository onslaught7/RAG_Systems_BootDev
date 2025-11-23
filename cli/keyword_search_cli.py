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
    try:    
        with open("./data/movies.json", "r") as f:
            movies_dict = json.load(f)
            
    except FileNotFoundError:
        print("Error: ./data/movies.json not found.")
        return []

    search_results = []
    normalized_query = _normalize_text(query)
    if not normalized_query:
        return []

    def _partial_match(query_tokens: List[str], title_tokens: List[str]) -> bool:
        """Check if any token in query_tokens partially matches any token in title_tokens."""
        return any(token in title_token for token in query_tokens for title_token in title_tokens)

    # Handle case where "movies" key might be missing or not a list .get() instead of direct access
    for movie in movies_dict.get("movies", []):
        movie_title_token = _normalize_text(movie.get("title", ""))
        if _partial_match(normalized_query, movie_title_token):
            search_results.append((movie["id"], movie["title"]))

    # Order by ascending order of IDs and truncate to top 5 results
    search_results.sort(key=lambda x: x[0])
    top_results = search_results[:5]

    return [title for _, title in top_results]


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
            docs = idx.get_documents('merida')
            if docs:
                print(f"First document for token 'merida' = {docs[0]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
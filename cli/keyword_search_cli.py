import argparse # To parse command-line arguments
from typing import List
import json
import string


def _normalize_text(text: str) -> str:
    """
    Noramlize text for keyword search. Perfroms the following operations:

    - Case in-sensitivity
    - Punctuation removal
    - Tokenization
    - Stop words removal
    - Stemming    
    """
    if not isinstance(text, str):
        return ""

    translator = str.maketrans("", "", string.punctuation) 
    cleaned = text.lower().translate(translator)
    
    return cleaned

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

    for movie in movies_dict["movies"]:
        if normalized_query in _normalize_text(movie["title"]):
            result = (movie["id"], movie["title"])
            search_results.append(result)

    # Order by ascending order of IDs and truncate to top 5 results
    search_results.sort(key=lambda x: x[0])
    top_results = search_results[:5]

    return [title for _, title in top_results]


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
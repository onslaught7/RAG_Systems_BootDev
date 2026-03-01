import json
import argparse
from lib.hybrid_search import HybridSearch


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("normalize", help="Accepts a list of scores and prints the normalized scores")
    search_parser.add_argument("scores", nargs="+", type=float)

    search_parser = subparsers.add_parser("weighted-search", help="Search with combined logic of semantic search and keyword search.")   
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--alpha", type=float, default=0.5, help="Configurabe parameter to go into the hybrid searh algo.")
    search_parser.add_argument("--limit", type=int, default=5, help="The number of query response.")


    args = parser.parse_args()

    
    def load_movies():
        with open("./data/movies.json", "r") as f:
            data = json.load(f)
            documents = data["movies"]
        return documents
        
    match args.command:
        case "normalize":
            documents = load_movies()
            hbd = HybridSearch(documents)

            hbd.min_max_normalization(args.scores)
        case "weighted-search":
            documents = load_movies()
            hbd = HybridSearch(documents)

            results = hbd.weighted_search(
                args.query,
                args.alpha,
                args.limit
            )

            for rank, result in enumerate(results, start=1):
                print(f"{rank}. {result['title']}")
                print(f"   Hybrid Score: {result['hybrid_score']:.3f}")
                print(f"   BM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f}")
                print(f"   {result['document']}")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
import json
import argparse
from lib.hybrid_search import HybridSearch
from enhance_with_llm import Gemini


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("normalize", help="Accepts a list of scores and prints the normalized scores")
    search_parser.add_argument("scores", nargs="+", type=float)

    search_parser = subparsers.add_parser("weighted-search", help="Search with combined logic of semantic search and keyword search.")   
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--alpha", type=float, default=0.5, help="Configurabe parameter to go into the hybrid searh algo.")
    search_parser.add_argument("--limit", type=int, default=5, help="The number of query response.")

    search_parser = subparsers.add_parser("rrf-search", help="Search using rankings instead of scores.")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("-k", type=int, default=60, help="Configurabe parameter to signify weight given to ranks.")
    search_parser.add_argument("--limit", type=int, default=5, help="Then number of response for the query input.")
    search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    search_parser.add_argument("--rerank-method", type=str, choices=["individual"], default="", help="Add reraking to improve search results for the query.")

    # Add a new optional string argument --rerank-method to the rrf-search command. For now it can either be "individual" or not supplied at all. We'll add more options later.
    # Perform rrf search as normal, but gather 5 times as many results as the "limit" specified if the --rerank-method is set to "individual".
    # If the "individual" rerank method is provided, after doing the initial rrf search, run the results through a series of LLM prompts (one per document) asking the LLM to provide a new score for each document. I used this system prompt:
    # f"""Rate how well this movie matches the search query.

    # Query: "{query}"
    # Movie: {doc.get("title", "")} - {doc.get("document", "")}

    # Consider:
    # - Direct relevance to query
    # - User intent (what they're looking for)
    # - Content appropriateness

    # Rate 0-10 (10 = perfect match).
    # Give me ONLY the number in your response, no other text or explanation.

    # Score:"""

    # To avoid hitting a rate limit, I recommend sleeping for 3 seconds (or longer if you're running into issues) between each LLM call.
    # Sort the results by the new score in descending order
    # Print the results after truncating to the limit, preserving the RRF score and the new LLM score in this format:
    # Reranking top 3 results using individual method...
    # Reciprocal Rank Fusion Results for 'family movie about bears in the woods' (k=60):
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
        case "rrf-search":
            documents = load_movies()
            hbd = HybridSearch(documents)

            query_to_use = args.query

            if args.enhance:
                enhanced_query = Gemini.enhance_search(args.query, args.enhance)
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n")
                query_to_use = enhanced_query

            if args.rerank_method == "individual":
                results = hbd.rrf_search(
                    query_to_use,
                    args.k,
                    args.limit * 5
                )
                enhanced_results = Gemini.enhanced_score(query_to_use, results)
                print(f"Reranking top {args.limit} results using individual method...")
                print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k}):")
                print()

                for rank, result in enumerate(enhanced_results[:args.limit], start=1):
                    print(f"{rank}. {result['title']}")
                    print(f"   Rerank Score: {result['rerank_score']:.3f}/10")
                    print(f"   RRF Score: {result['rrf_score']:.3f}")
                    print(f"   BM25 Rank: {int(result['bm25_rank'])}, Semantic Rank: {int(result['semantic_rank'])}")
                    print(f"   {result['document']}")
                    print()
            else:
                results = hbd.rrf_search(
                    query_to_use,
                    args.k,
                    args.limit
                )
                for rank, result in enumerate(results, start=1):
                    print(f"{rank}. {result['title']}")
                    print(f"   RRF SCORE: {result['rrf_score']:.3f}")
                    print(f"   BM25 Rank: {result['bm25_rank']:.3f}, Semantic Rank: {result['semantic_rank']:.3f}")
                    print(f"   {result['document']}")
                    print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
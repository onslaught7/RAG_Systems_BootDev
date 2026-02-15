import json
import argparse
from lib.semantic_search import (
    verify_model, 
    embed_text, 
    verify_embeddings, 
    embed_query_text,
    chunk_text,
    semantic_chunk_text,
    SemanticSearch,
    ChunkedSemanticSearch
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Get model and max sequence length.")

    search_parser = subparsers.add_parser("embed_text", help="Perform vector embedding on the input text.")
    search_parser.add_argument("embed_text", type=str, help="Get embedding for the text.")

    search_parser = subparsers.add_parser("verify_embeddings", help="Verify whether document has been embedded and create if hasn't.")

    search_parser = subparsers.add_parser("embedquery", help="Embedd query text")
    search_parser.add_argument("query", type=str, help="Embed query text.")

    search_parser = subparsers.add_parser("search", help="Search for movie using query")
    search_parser.add_argument("query", type=str, help="The name or context of the movie to be searched.")
    search_parser.add_argument("--limit", type=int, default=5, help="Return top limit results in descending order")

    search_parser = subparsers.add_parser("chunk", help="Break text to a specific number of chunks.")
    search_parser.add_argument("long_query", type=str, help="The input text to perform chunking on.")
    search_parser.add_argument("--chunk-size", type=int, default=200, help="The size of each chunk.")
    search_parser.add_argument("--overlap", type=int, help="The overlap size of the chunk.")

    search_parser = subparsers.add_parser("semantic_chunk", help="Break text to a specific number of chunks, while preserving the meaning of each chunk.")
    search_parser.add_argument("long_query", type=str, help="The input text to perform chunking on.")
    search_parser.add_argument("--max-chunk-size", type=int, default=4, help="The max size of each chunk, while preserving the meaning.")
    search_parser.add_argument("--overlap", type=int, default=0, help="The part of the chunk context to overlap onto the next.")

    search_parser = subparsers.add_parser("embed_chunks", help="Load the movie documents and build the chunk embeddings")

    search_parser = subparsers.add_parser("search_chunked", help="Search for matches using query")
    search_parser.add_argument("long_query", type=str, help="Query to search for.")
    search_parser.add_argument("--limit", type=int, default=5, help="Return the top matches. The number of top matches is determined by the limit")

    args = parser.parse_args()

    data_dir = "data/movies.json"
    def load_movies(path):
        with open(path, "r") as f:
            return json.load(f)["movies"]


    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.embed_text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            smn = SemanticSearch()
            with open(data_dir, "r") as f:
                data = json.load(f)
                documents = data["movies"]
                smn.load_or_create_embeddings(documents)

            result = smn.search(args.query, args.limit)
            for index, item in enumerate(result, start=1):
                print(f"{index}. {item['title']} (score: {item['score']:.4f})")
                print(f"{item['description']}\n")
        case "chunk":
            chunk_text(args.long_query, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_text(args.long_query, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            chunked_smn = ChunkedSemanticSearch()
            documents = load_movies(data_dir)

            embeddings = chunked_smn.load_or_create_chunk_embeddings(documents)

            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            chunked_smn = ChunkedSemanticSearch()
            documents = load_movies(data_dir)

            embeddings = chunked_smn.load_or_create_chunk_embeddings(documents)

            results = chunked_smn.search_chunks(args.long_query, args.limit)

            for i, result in enumerate(results, start=1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
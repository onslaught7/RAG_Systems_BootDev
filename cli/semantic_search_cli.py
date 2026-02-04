import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Get model and max sequence length.")

    search_parser = subparsers.add_parser("embed_text", help="Perform vector embedding on the input text.")
    search_parser.add_argument("embed_text", type=str, help="Get embedding for the text.")

    search_parser = subparsers.add_parser("verify_embeddings", help="Verify whether document has been embedded and create if hasn't.")
    
    args = parser.parse_args()


    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.embed_text)
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
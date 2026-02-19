import argparse
from lib.hybrid_search import (
    HybridSearch,
    min_max_normalization
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("normalize", help="Accepts a list of scores and prints the normalized scores")
    search_parser.add_argument("scores", nargs="+", type=float)
    


    args = parser.parse_args()

    match args.command:
        case "normalize":
            min_max_normalization(args.scores)                
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
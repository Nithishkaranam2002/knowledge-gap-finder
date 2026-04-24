import argparse
from pipeline import run_pipeline, print_results

def main():
    parser = argparse.ArgumentParser(description="Knowledge Gap Finder - CLI")
    parser.add_argument("query", type=str, help="Research topic to search")
    parser.add_argument("--limit", type=int, default=100, help="Number of papers to fetch")
    parser.add_argument("--top-k", type=int, default=10, help="Top K papers to retrieve")
    parser.add_argument("--refresh", action="store_true", help="Force refresh cache")
    args = parser.parse_args()

    results = run_pipeline(
        query=args.query,
        limit=args.limit,
        top_k=args.top_k,
        force_refresh=args.refresh
    )
    print_results(results)

if __name__ == "__main__":
    main()
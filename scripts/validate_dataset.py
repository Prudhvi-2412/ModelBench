"""Validate a JSONL dataset using the same schema as the API."""
import argparse
import sys

from modelbench_api.evaluation.datasets import load_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Dataset name without .jsonl")
    args = parser.parse_args()
    try:
        examples = load_dataset(args.name)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Invalid dataset: {exc}", file=sys.stderr)
        return 1
    print(f"Valid dataset '{args.name}': {len(examples)} examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

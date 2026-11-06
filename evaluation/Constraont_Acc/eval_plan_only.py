import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODEL_PATH, SAMPLE_PATHS, get_units_structured, load_rows, run  # noqa: E402

METHOD = "plan-only"


def get_units(row):
    return get_units_structured(row, "generation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=SAMPLE_PATHS[METHOD])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_rows(args.sample)
    result = run(rows, get_units, args.model, args.seed)
    result["method"] = METHOD
    result["sample"] = args.sample
    result["n_rows"] = len(rows)

    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

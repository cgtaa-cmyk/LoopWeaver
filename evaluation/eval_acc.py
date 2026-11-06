import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACC_DIR = HERE / "Constraont_Acc"
METHODS = [
    ("baseline", "eval_baseline.py"),
    ("gen-only", "eval_gen_only.py"),
    ("plan-only", "eval_plan_only.py"),
    ("loopweaver", "eval_loopweaver.py"),
]
KINDS = ("once", "range", "periodic", "Avg.")


def run_one(name, script, seed):
    proc = subprocess.run(
        [sys.executable, str(ACC_DIR / script), "--seed", str(seed)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return name, None, proc.stderr.strip()
    line = proc.stdout.strip().splitlines()[-1]
    return name, json.loads(line), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--method", choices=[m[0] for m in METHODS], default=None,
                    help="Evaluate only one method (default: all methods)")
    args = ap.parse_args()

    methods = METHODS if args.method is None else \
        [(name, script) for name, script in METHODS if name == args.method]

    rows = []
    for name, script in methods:
        name, summary, err = run_one(name, script, args.seed)
        if err is not None:
            print(f"[{name}] FAILED:\n{err}", file=sys.stderr)
            continue
        acc = summary["acc_strict"]
        row = {"method": name, "n_rows": summary["n_rows"],
               "n_prompts": summary["n_prompts"]}
        row.update(acc)
        rows.append(row)
        print(f"[{name}] done: n={summary['n_rows']} "
              f"prompts={summary['n_prompts']}", file=sys.stderr)

    if not rows:
        print("no method succeeded", file=sys.stderr)
        sys.exit(1)

    print("\n===== Constraint Accuracy (strict, single pass) =====")
    print(f"{'method':<12}" + "".join(f"{k:>12}" for k in KINDS))
    for row in rows:
        print(f"{row['method']:<12}" + "".join(f"{row[k]:>12}" for k in KINDS))
    print("\nJSON:")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

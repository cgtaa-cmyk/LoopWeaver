import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "Constraont_Acc"))
from common import SAMPLE_PATHS, load_rows  

RUBRIC = HERE / "Text_Quality" / "judge.txt"
ENDPOINT = ""
MODEL = "gpt-4o-mini"
TEMPERATURE = 0
MAX_TOKENS = 256
TIMEOUT = 180
MAX_ATTEMPTS = 4
DIMS = ("Narrative Continuity", "Entity & Memory Persistence",
        "Temporal Groundedness", "Affective Consistency")
METHODS = ("baseline", "gen-only", "plan-only", "loopweaver")
DEFAULT_KEY_FILE = "/path/to/.judge_key"


def read_key(key_file):
    key = os.environ.get("JUDGE_API_KEY", "").strip()
    if not key and key_file and Path(key_file).exists():
        key = Path(key_file).read_text().strip()
    if not key or any(c.isspace() for c in key):
        raise ValueError("no API key: set JUDGE_API_KEY or provide --key-file")
    return key


def request_payload(rubric, inst, response):
    content = re.sub(r"\$INST\$|\$RESPONSE\$",
                     lambda m: inst if m.group() == "$INST$" else response, rubric)
    return {"model": MODEL, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": content}]}


def parse_judgment(text):
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text,
                          flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    obj = json.loads(text)
    if not isinstance(obj, dict) or set(obj) != set(DIMS):
        raise ValueError("judge JSON must contain exactly the four dimension keys")
    for dim in DIMS:
        if type(obj[dim]) is not int or not 1 <= obj[dim] <= 5:
            raise ValueError(dim + ": score must be integer 1..5")
    return obj


def to_row(scores):
    converted = {dim: (scores[dim] - 1) * 25 for dim in DIMS}
    converted["Overall"] = statistics.mean(converted[dim] for dim in DIMS)
    return converted


def judge_one(payload, key):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        req = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + key})
        try:
            with opener.open(req, timeout=TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
            envelope = json.loads(raw)
            content = envelope["choices"][0]["message"]["content"]
            return parse_judgment(content)
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            low = raw.lower()
            if "quota_not_enough" in low or "insufficient balance" in low:
                raise RuntimeError("quota exhausted: " + raw[:200])
            last = "HTTP %d: %s" % (e.code, raw[:200])
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(1.0 * attempt)
    raise RuntimeError("judge failed after retries: " + (last or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-file", default=DEFAULT_KEY_FILE)
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--limit", type=int, default=0, help="Only test the first N samples (0=all)")
    ap.add_argument("--out", default=None, help="Save per-prompt scores to JSON")
    args = ap.parse_args()

    rubric = RUBRIC.read_text(encoding="utf-8")
    key = read_key(args.key_file)

    report = {}
    all_scores = {}
    for method in args.methods:
        rows = load_rows(SAMPLE_PATHS[method])
        if args.limit:
            rows = rows[: args.limit]

        per_prompt = []
        for row in rows:
            payload = request_payload(rubric, row["prompt"], row["final_text"])
            per_prompt.append(to_row(judge_one(payload, key)))
        all_scores[method] = per_prompt

        cols = list(DIMS) + ["Overall"]
        mean = {c: round(statistics.mean(r[c] for r in per_prompt), 2) for c in cols}
        report[method] = {"n": len(rows), "mean": mean}
        print(f"[{method}] done: n={len(rows)}", file=sys.stderr, flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(all_scores, ensure_ascii=False, indent=2))

    print("\n===== Text Quality =====")
    cols = list(DIMS) + ["Overall"]
    print(f"{'method':<12}" + "".join(f"{c:>30}" for c in cols))
    for method in args.methods:
        m = report[method]
        print(f"{method:<12}" + "".join(f"{m['mean'][c]:>8.2f}" for c in cols))
    print("\nJSON:")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

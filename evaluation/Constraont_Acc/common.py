import json
import re

from vllm import LLM, SamplingParams

KINDS = ("once", "range", "periodic")
MODEL_PATH = "/Qwen2.5-7B-Instruct"

FIELD = {"Week": "weekly_plan", "Menu Week": "weekly_plan",
         "Floor": "floor_plan", "Block": "block_plan"}
ID_KEY = {"Week": "week_id", "Menu Week": "week_id",
          "Floor": "floor_id", "Block": "block_id"}
BODY = {
    "planning":   {"Week": "events", "Menu Week": "dishes",
                   "Floor": "purpose", "Block": "use"},
    "generation": {"Week": "diary_entry", "Menu Week": "week_menu",
                   "Floor": "plan", "Block": "plan"},
}

SAMPLE_PATHS = {
    "baseline": "",
    "gen-only": "",
    "plan-only": "",
    "loopweaver": "",
}


def load_rows(path):
    text = open(path, encoding="utf-8").read().lstrip()
    if text.startswith("["):
        return json.loads(text)
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def get_units_structured(row, stage="both"):
    typ = row["type"]
    field, id_key = FIELD[typ], ID_KEY[typ]
    body_keys = ["planning", "generation"] if stage == "both" else [stage]
    out = {}
    for u in (row.get(field) or []):
        if not isinstance(u, dict):
            continue
        parts = []
        for bk in body_keys:
            b = u.get(BODY[bk][typ])
            if isinstance(b, str) and b.strip():
                parts.append(b.strip())
        if not parts:
            continue
        ident = str(u.get(id_key, ""))
        m = re.search(re.escape(typ) + r"\s+(\d+)\b", ident, re.I) or \
            re.fullmatch(r"\s*(\d+)\s*", ident)
        if m:
            n = int(m.group(1))
            if n not in out:
                out[n] = f"{typ} {n}: " + " | ".join(parts)
    return out


def get_units_output_blocks(row):
    typ = row["type"]
    out = {}
    for block in (row.get("output_blocks") or []):
        m = re.search(re.escape(typ) + r"\s+(\d+)\b", str(block), re.I)
        if m:
            n = int(m.group(1))
            if n not in out:
                out[n] = str(block)
    return out


def build_prompts(rows, get_units):
    prompts, refs = [], []
    for ri, row in enumerate(rows):
        units = get_units(row)
        for kind in KINDS:
            for ident, desc in row["checks_" + kind].items():
                n = int(ident)
                if n in units:
                    prompts.append(
                        f"Context: {units[n]}### Instruction: Does the context above "
                        f"include the {desc}? Please answer with 'yes' or 'no' only."
                    )
                    refs.append((ri, kind, ident))
    return prompts, refs


def strict_acc(rows, labels):
    expected = {k: sum(len(r["checks_" + k]) for r in rows) for k in KINDS}
    yes = {k: 0 for k in KINDS}
    for x in labels:
        if x["label"] == "yes":
            yes[x["kind"]] += 1
    return {k: round(100.0 * yes[k] / expected[k], 2) for k in KINDS}


def run(rows, get_units, model_path, seed=42):
    llm = LLM(model=model_path, tensor_parallel_size=1)
    prompts, refs = build_prompts(rows, get_units)
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.95, top_p=0.95, max_tokens=10, seed=seed),
    )
    labels = [
        {"kind": kind, "label": ("yes" if "yes" in o.outputs[0].text.strip().lower() else "no")}
        for o, (ri, kind, ident) in zip(outputs, refs)
    ]
    acc = strict_acc(rows, labels)
    acc["Avg."] = round(sum(acc[k] for k in KINDS) / 3, 2)
    return {
        "n_prompts": len(prompts),
        "expected": {k: sum(len(r["checks_" + k]) for r in rows) for k in KINDS},
        "acc_strict": acc,
    }

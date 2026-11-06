#!/usr/bin/env python3
"""
QualityRewardAgent — a rule-based quality reward for DPO self-distillation.

Motivation: the old RewardAgent's reward was dominated by length bias (keyword hit count + constraint coverage + length),
which made the model "longer but worse" after DPO training. This agent replaces that signal with four pure rule-based
quality proxies, with no dependence on any judge / teacher model, and without modifying RewardAgent.py.

Four quality proxies (all computable within an example, output at step level):
  1. substantive  min(1, content word count / 100)   (empty/short steps score low, preventing degeneration to empty output)
  2. constraint   tightened constraint hit count / total expected constraints   (higher means more on-topic, independent of word count)
  3. anti_rep     1 - local window duplicate bigram ratio   (windowed, independent of length)
  4. distinct     1 - average word-frequency cosine similarity to other steps in the same example (length-insensitive)

Combination: z-score the four raw metrics within the example, weighted sum, then min-max to [0,1].

Note: the old density (herdan = log(unique words)/log(total words)) was removed — it is naturally higher for short text,
treating "concise and short" as "high quality", which was the main cause of the model degenerating into empty output after
DPO self-distillation. constraint was also changed from "hits/word count" to "hits/expected count" to remove length bias.
anti_rep was changed to a local duplicate rate within a fixed window; distinct was changed from Jaccard to word-frequency
cosine similarity (Jaccard is low for short text, so short steps were misjudged as "highly distinct"), and neither rewards
short text anymore. A new substantive proxy (content word count / 100, capped) was added as the main signal, explicitly
giving empty/short steps a low score — otherwise anti_rep/distinct would still pick the "shortest step" as chosen, and DPO
would keep learning "shorter is better". A completeness gate is also added before rollout: early-stopped examples
(coverage < 50%) are skipped directly and produce no DPO signal.

The output contract is identical to RolloutRewardAgent.rollout_rewards:
  - each step gets step["reward"] (float) and step["advantage"] (float)
  - the example gets max_reward_steps / min_reward_steps / reward_stats
so it can be consumed directly by downstream DPODataAgent, only the scoring source needs to change.
"""
import math
import re
import logging
from collections import Counter

BODY_FIELD = {"Week": "diary_entry", "Menu Week": "week_menu", "Floor": "plan", "Block": "plan"}
WEIGHTS = {"substantive": 0.45, "constraint": 0.30, "anti_rep": 0.15, "distinct": 0.10}
SUBST_TARGET = 100

_STOP = set(
    "a an the of to in and or for on at with by from as is are was were be been "
    "this that these those it its i you he she they we our your his her their "
    "my me him them not no but if then than so very just also do does did have has had".split()
)


def _words(text):
    return [w for w in re.findall(r"[a-zA-Z]+", str(text).lower()) if w not in _STOP]


def _herdan(text):
    """Log-corrected type-token ratio, a length-insensitive information density."""
    ws = _words(text)
    if len(ws) < 2:
        return 0.0
    uniq = len(set(ws))
    if uniq < 2:
        return 0.0
    return math.log(uniq) / math.log(len(ws))


_REP_WINDOW = 40


def _rep_bigram(text, window=_REP_WINDOW):
    """Local duplicate rate: compute the duplicate bigram ratio within a fixed sliding window, take the max.

    A fixed window is used instead of the full text because the global duplicate rate naturally rises with length
    (longer text is more likely to hit duplicate bigrams), which would misjudge "long" as "padding". After windowing
    it is length-independent: a long text is not penalized as long as it does not repeat locally.
    """
    ws = _words(text)
    if len(ws) < 2:
        return 0.0
    if len(ws) <= window:
        bigrams = list(zip(ws, ws[1:]))
        dup = sum(1 for b, c in Counter(bigrams).items() if c > 1)
        return dup / len(bigrams)
    best = 0.0
    for i in range(0, len(ws) - window + 1, max(1, window // 2)):
        w = ws[i:i + window]
        bg = list(zip(w, w[1:]))
        dup = sum(1 for b, c in Counter(bg).items() if c > 1)
        best = max(best, dup / len(bg))
    return best


def _cosine(ca, cb):
    """Word-frequency cosine similarity (0~1), length-insensitive."""
    if not ca or not cb:
        return 0.0
    if len(ca) <= len(cb):
        dot = sum(v * cb.get(k, 0) for k, v in ca.items())
    else:
        dot = sum(v * ca.get(k, 0) for k, v in cb.items())
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _constraint_hits(word_set, expected):
    """Tightened constraint hit test: >50% of the non-stopwords (with len>3) in expected must appear in the body to count as a hit.

    The old _soft_match's `pattern in text` and "any single word hit" are overly permissive for long text and degrade
    into a length proxy; here most constraint words must appear together to count as a real hit.
    """
    if not expected:
        return 0
    exp_words = [w for w in re.findall(r"[a-zA-Z]+", str(expected).lower()) if w not in _STOP and len(w) > 3]
    if not exp_words:
        return 0
    hits = sum(1 for w in exp_words if w in word_set)
    return 1 if hits / len(exp_words) > 0.5 else 0


class QualityRewardAgent:
    """Rule-based quality reward rollout scoring. Interface compatible with RolloutRewardAgent.rollout_rewards."""

    @staticmethod
    def _kind(example):
        t = (example.get("type") or "").lower()
        if "floor" in t:
            return "Floor"
        if "block" in t:
            return "Block"
        if "menu" in t:
            return "Menu Week"
        return "Week"

    @staticmethod
    def _body_of(step, kind):
        if not isinstance(step, dict):
            return str(step)
        f = BODY_FIELD.get(kind, "plan")
        v = step.get(f, "")
        return v if isinstance(v, str) else str(v)

    @staticmethod
    def _all_expects(example):
        expects = []
        for key in ("checks_once", "checks_range", "checks_periodic"):
            d = example.get(key) or {}
            if isinstance(d, dict):
                for v in d.values():
                    if isinstance(v, str):
                        expects.append(v)
                    elif isinstance(v, list):
                        expects += [x for x in v if isinstance(x, str)]
        return expects

    @classmethod
    def rollout_rewards(cls, example):
        """Compute quality rewards step by step for valid sub-plans; the output contract matches the old RewardAgent."""
        if not (example.get("is_valid") or str(example.get("filter_response", "")).upper() == "YES"):
            logging.info(f"Example {example.get('id', '')} skipped: not valid.")
            return example

        plan_key = None
        for key in ("weekly_plan", "floor_plan", "block_plan",
                    "filtered_weekly_plan", "filtered_floor_plan", "filtered_block_plan"):
            if key in example and isinstance(example[key], list) and example[key]:
                plan_key = key
                break
        if not plan_key:
            logging.warning(f"Example {example.get('id', '')} has no valid plan to rollout.")
            return example

        plan = example[plan_key]
        if plan_key.startswith("filtered_"):
            if isinstance(plan, dict) and "original_plan" in plan:
                plan = plan["original_plan"]
        if not isinstance(plan, list):
            logging.warning(f"Plan for {example.get('id', '')} is not a list.")
            return example
        kind = cls._kind(example)
        example['generation_reward_is_diagnostic_only'] = True
        example['generation_dpo_requires_same_request_review'] = True
        expects = cls._all_expects(example)

        try:
            number = int(example.get("number") or (100 if kind in ("Floor", "Block") else 52))
        except (TypeError, ValueError):
            number = 100 if kind in ("Floor", "Block") else 52
        if len(plan) < number * 0.5:
            logging.warning(
                f"Example {example.get('id', '')} early-stop "
                f"(steps={len(plan)}/{number}), skip reward rollout."
            )
            return example

        bodies = []
        word_sets = []
        for step in plan:
            if isinstance(step, str):
                step = {"content": step}
            b = cls._body_of(step, kind)
            ws = _words(b)
            bodies.append(b)
            word_sets.append(ws)

        n = len(bodies)

        substantives = [min(1.0, len(ws) / SUBST_TARGET) for ws in word_sets]
        anti_reps = [1.0 - _rep_bigram(b) for b in bodies]
        constraints = []
        for ws, b in zip(word_sets, bodies):
            hits = sum(_constraint_hits(set(ws), e) for e in expects) if expects else 0
            constraints.append(hits / len(expects) if expects else 0.0)
        counters = [Counter(ws) for ws in word_sets]
        distincts = []
        for i in range(n):
            if n < 2:
                distincts.append(0.0)
            else:
                sims = [_cosine(counters[i], counters[j]) for j in range(n) if j != i]
                distincts.append(1.0 - (sum(sims) / len(sims) if sims else 0.0))

        raw = {
            "substantive": substantives,
            "constraint": constraints,
            "anti_rep": anti_reps,
            "distinct": distincts,
        }

        def zscore(vals):
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / len(vals)
            std = math.sqrt(var) if var > 1e-9 else 1.0
            return [(v - m) / std for v in vals]

        z = {k: zscore(v) for k, v in raw.items()}
        qualities = [
            sum(WEIGHTS[k] * z[k][i] for k in WEIGHTS)
            for i in range(n)
        ]

        qmin, qmax = min(qualities), max(qualities)
        if qmax - qmin < 1e-9:
            rewards = [0.5] * n
        else:
            rewards = [(q - qmin) / (qmax - qmin) for q in qualities]

        processed = []
        for step, r in zip(plan, rewards):
            if isinstance(step, str):
                step = {"content": step}
            step = dict(step)
            step["reward"] = float(r)
            processed.append(step)
        plan = processed

        max_reward = max(rewards)
        min_reward = min(rewards)
        max_steps = [s for s in plan if s.get("reward") == max_reward]
        min_steps = [s for s in plan if s.get("reward") == min_reward]

        arr = list(rewards)
        mean = sum(arr) / len(arr)
        var = sum((v - mean) ** 2 for v in arr) / len(arr)
        std = math.sqrt(var) if var > 1e-9 else 1.0
        for s in plan:
            s["advantage"] = float((s.get("reward", 0) - mean) / std)

        example[plan_key] = plan
        example["max_reward_steps"] = max_steps
        example["min_reward_steps"] = min_steps
        example["reward_stats"] = {
            "mean": mean,
            "std": std,
            "max": float(max_reward),
            "min": float(min_reward),
            "positive_rewards": sum(1 for r in rewards if r > 0),
            "total_steps": len(rewards),
            "using_fallback": False,
            "reward_source": "quality_rule",
        }
        return example

    @staticmethod
    async def async_rollout_rewards(model, example, semaphore):
        return QualityRewardAgent.rollout_rewards(example)

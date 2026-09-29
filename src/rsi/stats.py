"""Statistics of a self-improvement run: coverage, churn and harvest difficulty.

    python src/rsi/stats.py --train_data data/raw/chebi-20/train.txt \
        data/MoIA/T0.json data/MoIA/T1.json data/MoIA/T2.json

Entries without an "id" (the conversation-format seed) are matched to training
ids by description and reference molecule.

For a harvest, pass@k of the sampling policy on the training set is the share of
instances whose first verified sample is among the first k (failure_times < k);
samples are independent, so this estimates pass@k for every k up to the budget.
Traces carried over from an earlier set are left out.
"""

import argparse
import collections
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import dump_json, load_json, load_tsv, parse_entry, to_inchi


def resolve_ids(entries, train_items):
    """Training id for each entry (None if it cannot be matched unambiguously)."""
    by_q = collections.defaultdict(list)
    for it in train_items:
        by_q[it["question"]].append(it)
    ids = []
    for d in entries:
        if d.get("id"):
            ids.append(str(d["id"]))
            continue
        question, gt, _ = parse_entry(d)
        cands = by_q.get(question, [])
        if len(cands) > 1:
            gi = to_inchi(gt)
            cands = [c for c in cands if gi is not None and to_inchi(c["gt"]) == gi]
        ids.append(cands[0]["id"] if len(cands) == 1 else None)
    return ids


def summarize(entries, ids, universe_size):
    unique = {i for i in ids if i is not None}
    ft = [d["failure_times"] for d in entries if "failure_times" in d]
    lengths = [len(parse_entry(d)[2]) for d in entries]
    out = {
        "rows": len(entries),
        "unique_ids": len(unique),
        "unresolved_rows": sum(1 for i in ids if i is None),
        "duplicate_rows": len(entries) - len(unique) - sum(1 for i in ids if i is None),
        "coverage": len(unique) / universe_size,
        "mean_trace_chars": statistics.mean(lengths) if lengths else 0.0,
    }
    if ft:
        out.update({
            "first_try_rate": sum(1 for f in ft if f == 0) / len(ft),
            "mean_failure_times": statistics.mean(ft),
            "max_failure_times": max(ft),
        })
    first_success = {}
    for d, i in zip(entries, ids):
        if i is not None and "failure_times" in d and not d.get("from_previous"):
            first_success[i] = min(d["failure_times"], first_success.get(i, d["failure_times"]))
    if first_success:
        budget = 1
        while budget <= max(first_success.values()):
            budget *= 2
        out["pass_at_k"] = {k: sum(1 for f in first_success.values() if f < k) / universe_size
                            for k in (2 ** j for j in range(budget.bit_length()))}
    return out, unique


def compare(prev_ids, curr_ids):
    new, lost = curr_ids - prev_ids, prev_ids - curr_ids
    return {
        "new": len(new), "lost": len(lost), "retained": len(curr_ids & prev_ids),
        "net_growth": (len(curr_ids) - len(prev_ids)) / len(prev_ids) if prev_ids else None,
    }


def main():
    p = argparse.ArgumentParser(description="Coverage and churn across trace sets R^0..R^T")
    p.add_argument("--train_data", required=True)
    p.add_argument("trace_sets", nargs="+", help="Trace-set JSON files in iteration order")
    p.add_argument("--json_out", default=None)
    args = p.parse_args()

    train = load_tsv(args.train_data)
    rows, prev = [], None
    for t, path in enumerate(args.trace_sets):
        entries = load_json(path)
        summary, ids = summarize(entries, resolve_ids(entries, train), len(train))
        summary.update({"iteration": t, "path": path})
        if prev is not None:
            summary.update(compare(prev, ids))
        rows.append(summary)
        prev = ids

    head = f"{'T':>2} {'rows':>6} {'unique':>6} {'cover':>6} {'new':>6} {'lost':>5} {'retain':>6} {'1st-try':>7} {'mean ft':>7} {'chars':>6}"
    print(head)
    for r in rows:
        def cell(k, width, spec=""):
            return format(r[k], f">{width}{spec}") if r.get(k) is not None else "-".rjust(width)
        print(f"{r['iteration']:>2} {r['rows']:>6} {r['unique_ids']:>6} {r['coverage']:>6.1%} "
              f"{cell('new', 6)} {cell('lost', 5)} {cell('retained', 6)} "
              f"{cell('first_try_rate', 7, '.1%')} {cell('mean_failure_times', 7, '.2f')} {r['mean_trace_chars']:>6.0f}")
        if r.get("pass_at_k"):
            print("   pass@k: " + "  ".join(f"{k}: {v:.1%}" for k, v in r["pass_at_k"].items()))
        if r["unresolved_rows"]:
            print(f"   ({r['unresolved_rows']} rows could not be matched to a training id)")
    if args.json_out:
        dump_json(rows, args.json_out)


if __name__ == "__main__":
    main()

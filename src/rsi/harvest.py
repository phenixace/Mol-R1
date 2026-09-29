"""Harvest operator of the self-improvement loop.

The current policy samples reasoning traces for every training instance, in rounds
of `samples_per_round`, until one trace is verified (well-formed and the final
answer is the reference molecule) or `max_attempts` samples have been drawn. The
first verified trace is kept. No teacher model is involved. Prompts use the chat
format of src.common.chat_prompt, as in SFT, RL and evaluation.

Output entries follow the released trace-set schema:
  id, question, gt, content, pred, score (=1), failure_times
where failure_times is the number of failed samples before the accepted one.

Run one shard per GPU, then merge them with --merge (scripts/run_harvest.sh does
both; src/rsi/loop.py calls the same functions).
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import dump_json, load_items, load_json, verify


def harvest(items, sample_fn, max_attempts=8, samples_per_round=8,
            require_format=True, log=print):
    """Return (accepted_entries, meta) for items = [{id, question, gt}, ...].

    sample_fn(questions, n) -> n sampled responses per description.
    """
    accepted = {}
    pending = list(items)
    used = 0
    rounds = 0
    samples_generated = 0
    while pending and used < max_attempts:
        n = min(samples_per_round, max_attempts - used)
        completions = sample_fn([it["question"] for it in pending], n)
        samples_generated += n * len(pending)
        still_pending = []
        for it, comps in zip(pending, completions):
            for j, content in enumerate(comps):
                ok, pred = verify(content, it["gt"], require_format)
                if ok:
                    accepted[it["id"]] = {
                        "id": it["id"], "question": it["question"], "gt": it["gt"],
                        "content": content, "pred": pred, "score": 1,
                        "failure_times": used + j,
                    }
                    break
            else:
                still_pending.append(it)
        used += n
        rounds += 1
        pending = still_pending
        log(f"[harvest] round {rounds}: attempts {used}/{max_attempts}, "
            f"accepted {len(accepted)}/{len(items)}, pending {len(pending)}")

    entries = [accepted[it["id"]] for it in items if it["id"] in accepted]
    meta = {
        "items": len(items), "accepted": len(entries), "unresolved": len(pending),
        "samples_generated": samples_generated, "rounds": rounds,
        "max_attempts": max_attempts, "samples_per_round": samples_per_round,
        "require_format": require_format,
    }
    return entries, meta


def shard_paths(out_dir, shard_id):
    return (os.path.join(out_dir, f"shard_{shard_id}.json"),
            os.path.join(out_dir, f"shard_{shard_id}.meta.json"))


def merge_shards(out_dir, num_shards, items, output_path):
    """Merge shard outputs into one trace set ordered like `items`; return meta."""
    by_id, meta = {}, {"items": 0, "accepted": 0, "unresolved": 0, "samples_generated": 0}
    for i in range(num_shards):
        entries_path, meta_path = shard_paths(out_dir, i)
        for e in load_json(entries_path):
            by_id[e["id"]] = e
        m = load_json(meta_path)
        for k in meta:
            meta[k] += m[k]
        for k in ("max_attempts", "samples_per_round", "require_format"):
            meta[k] = m[k]
    entries = [by_id[it["id"]] for it in items if it["id"] in by_id]
    dump_json(entries, output_path)
    return meta


def main():
    p = argparse.ArgumentParser(description="Harvest verified self-generated reasoning traces")
    p.add_argument("--model_dir", help="Policy to sample from (not needed with --merge)")
    p.add_argument("--train_data", required=True, help="Raw training split (TSV) or JSON")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_attempts", type=int, default=8)
    p.add_argument("--samples_per_round", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--no_require_format", action="store_true",
                   help="Accept a correct answer even if the trace is not <think>/<answer> formatted")
    p.add_argument("--merge", action="store_true",
                   help="Merge the --num_shards shards in --output_dir into harvest.json and "
                        "harvest_meta.json instead of sampling")
    args = p.parse_args()

    if args.merge:
        meta = merge_shards(args.output_dir, args.num_shards, load_items(args.train_data),
                            os.path.join(args.output_dir, "harvest.json"))
        dump_json(meta, os.path.join(args.output_dir, "harvest_meta.json"))
        print(f"[harvest] merged {args.num_shards} shards: {meta['accepted']}/{meta['items']} "
              f"accepted, {meta['samples_generated']} samples")
        return
    if not args.model_dir:
        p.error("--model_dir is required unless --merge is given")

    from src.rsi.sampler import VLLMSampler

    items = load_items(args.train_data)[args.shard_id::args.num_shards]
    sampler = VLLMSampler(args.model_dir, temperature=args.temperature, top_p=args.top_p,
                          max_tokens=args.max_tokens, seed=args.seed + args.shard_id,
                          tensor_parallel_size=args.tensor_parallel_size,
                          gpu_memory_utilization=args.gpu_memory_utilization)
    entries, meta = harvest(items, sampler, args.max_attempts, args.samples_per_round,
                            require_format=not args.no_require_format)
    os.makedirs(args.output_dir, exist_ok=True)
    entries_path, meta_path = shard_paths(args.output_dir, args.shard_id)
    dump_json(entries, entries_path)
    dump_json(meta, meta_path)


if __name__ == "__main__":
    main()

"""Generate test-set predictions with a trained checkpoint (one data shard per GPU under torchrun).

Prompts use the chat format of src.common.chat_prompt, as in training; the prediction is
the last <answer> block of each output (src.common.extract_answer).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import chat_prompt, extract_answer, load_tsv, same_molecule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="predictions.txt")
    parser.add_argument("--test_path", type=str, default="data/raw/chebi-20/test.txt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=10000)
    parser.add_argument("--save_generations", type=str, default=None,
                        help="Also write full generations (JSONL) to this path")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group("nccl")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    items = load_tsv(args.test_path)
    all_cids = [it["id"] for it in items]
    all_gt = [it["gt"] for it in items]
    all_prompts = [chat_prompt(tokenizer, it["question"]) for it in items]

    # Shard data across GPUs
    my_indices = list(range(local_rank, len(all_prompts), world_size))
    my_prompts = [all_prompts[i] for i in my_indices]
    my_cids = [all_cids[i] for i in my_indices]
    my_gt = [all_gt[i] for i in my_indices]

    my_predictions, my_generations = [], []
    pbar = tqdm(range(0, len(my_prompts), args.batch_size),
                desc=f"GPU {local_rank}", disable=(local_rank != 0))

    for i in pbar:
        batch = my_prompts[i:i + args.batch_size]
        # the chat template already contains the BOS token
        inputs = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            do_sample = args.temperature > 0
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=args.max_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            if do_sample:
                gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
            else:
                gen_kwargs.update(do_sample=False)
            outputs = model.generate(**gen_kwargs)

        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            gen_ids = output[input_len:]
            gen_ids = gen_ids[(gen_ids != tokenizer.pad_token_id)]
            generated = tokenizer.decode(gen_ids, skip_special_tokens=False)
            my_predictions.append(extract_answer(generated))
            my_generations.append(generated)

    # Gather results from all GPUs
    if world_size > 1:
        all_results = [None] * world_size
        my_data = list(zip(my_indices, my_cids, my_gt, my_predictions, my_generations))
        dist.all_gather_object(all_results, my_data)

        if local_rank == 0:
            merged = []
            for rank_data in all_results:
                merged.extend(rank_data)
            merged.sort(key=lambda x: x[0])
            final_cids = [x[1] for x in merged]
            final_gt = [x[2] for x in merged]
            final_preds = [x[3] for x in merged]
            final_gens = [x[4] for x in merged]
        else:
            final_cids, final_gt, final_preds, final_gens = [], [], [], []
    else:
        final_cids, final_gt, final_preds, final_gens = my_cids, my_gt, my_predictions, my_generations

    if local_rank == 0:
        with open(args.output_path, "w", encoding="utf-8") as f:
            f.write("CID\tground_truth\tprediction\n")
            for cid, gt, pred in zip(final_cids, final_gt, final_preds):
                f.write(f"{cid}\t{gt}\t{pred}\n")

        if args.save_generations:
            with open(args.save_generations, "w", encoding="utf-8") as f:
                for cid, gt, pred, gen in zip(final_cids, final_gt, final_preds, final_gens):
                    f.write(json.dumps({"id": cid, "gt": gt, "prediction": pred, "generation": gen},
                                       ensure_ascii=False) + "\n")

        exact = sum(1 for g, p in zip(final_gt, final_preds) if same_molecule(p, g))
        total = len(final_preds)
        print(f"\nPredictions saved to {args.output_path}")
        print(f"Total: {total}, Exact match: {exact}/{total} = {exact/total*100:.2f}%")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

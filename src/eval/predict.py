import argparse
import csv
import re
import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

COT_PROMPT = "Based on the description, generate the SMILES of the molecule with reasoning.\nDescription: {question}\n\n<think>"
GT_PROMPT = "Based on the description, generate the SMILES of the molecule.\nDescription: {question}\n\n[START_I_SMILES]"


def extract_smiles_cot(text):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"</think>\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip().split("\n")[-1].strip()


def extract_smiles_gt(text):
    return text.replace("[END_I_SMILES]", "").replace("</s>", "").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cot", "gt"], required=True)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default="data/chebi-20/test.txt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=10000)
    args = parser.parse_args()

    if args.model_dir is None:
        args.model_dir = f"sft_output_{args.mode}/final"
    if args.output_path is None:
        args.output_path = f"predictions_{args.mode}.txt"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group("nccl")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    is_galactica = "galactica" in args.model_dir.lower()
    if is_galactica:
        tokenizer.pad_token_id = 1
        tokenizer.eos_token_id = 2
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    if args.mode == "cot":
        prompt_tpl = COT_PROMPT
        max_new = args.max_tokens
        extract_fn = extract_smiles_cot
    else:
        prompt_tpl = GT_PROMPT
        max_new = 256
        extract_fn = extract_smiles_gt
        end_token_id = tokenizer.encode("[END_I_SMILES]", add_special_tokens=False)[0]

    all_cids, all_gt, all_prompts = [], [], []
    with open(args.test_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            all_cids.append(row[0])
            all_gt.append(row[1])
            all_prompts.append(prompt_tpl.format(question=row[2]))

    # Shard data across GPUs
    my_indices = list(range(local_rank, len(all_prompts), world_size))
    my_prompts = [all_prompts[i] for i in my_indices]
    my_cids = [all_cids[i] for i in my_indices]
    my_gt = [all_gt[i] for i in my_indices]

    my_predictions = []
    pbar = tqdm(range(0, len(my_prompts), args.batch_size),
                desc=f"GPU {local_rank}", disable=(local_rank != 0))

    for i in pbar:
        batch = my_prompts[i:i + args.batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512,
        ).to(device)
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            eos_ids = [end_token_id, tokenizer.eos_token_id] if args.mode == "gt" else [tokenizer.eos_token_id]
            do_sample = args.temperature > 0
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=max_new,
                eos_token_id=eos_ids,
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
            my_predictions.append(extract_fn(generated))

    # Gather results from all GPUs
    if world_size > 1:
        all_results = [None] * world_size
        my_data = list(zip(my_indices, my_cids, my_gt, my_predictions))
        dist.all_gather_object(all_results, my_data)

        if local_rank == 0:
            merged = []
            for rank_data in all_results:
                merged.extend(rank_data)
            merged.sort(key=lambda x: x[0])
            final_cids = [x[1] for x in merged]
            final_gt = [x[2] for x in merged]
            final_preds = [x[3] for x in merged]
        else:
            final_cids, final_gt, final_preds = [], [], []
    else:
        final_cids, final_gt, final_preds = my_cids, my_gt, my_predictions

    if local_rank == 0:
        with open(args.output_path, "w", encoding="utf-8") as f:
            f.write("CID\tground_truth\tprediction\n")
            for cid, gt, pred in zip(final_cids, final_gt, final_preds):
                f.write(f"{cid}\t{gt}\t{pred}\n")

        exact = sum(1 for g, p in zip(final_gt, final_preds) if g.strip() == p.strip())
        total = len(final_preds)
        print(f"\n[{args.mode.upper()}] Predictions saved to {args.output_path}")
        print(f"[{args.mode.upper()}] Total: {total}, Exact match: {exact}/{total} = {exact/total*100:.2f}%")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

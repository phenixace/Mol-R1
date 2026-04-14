"""MoIA (Molecular Iterative Adaptation): SFT → RPO → Rejection-Sampling Distillation → repeat.

This script orchestrates the full MoIA training loop:
  1. SFT cold-start on PRID/RS distilled data
  2. RPO (Reinforced Policy Optimization) on the SFT model
  3. Use the RPO model to generate predictions on raw training data
  4. Rejection-sampling distillation: score predictions, keep correct ones as new data
  5. Merge new data into the training set
  6. Repeat from step 1 until data stops growing or max steps reached
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, desc=""):
    """Run a shell command and stream output."""
    print(f"\n{'='*60}")
    print(f"[STEP] {desc}")
    print(f"[CMD]  {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, check=True)
    return result.returncode


def count_json_entries(path):
    """Count entries in a JSON file."""
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return len(data)


def merge_distilled_data(existing_path, new_path, output_path):
    """Merge new distilled data into existing dataset, deduplicating by ID."""
    existing = []
    if os.path.exists(existing_path):
        with open(existing_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    new_data = []
    if os.path.exists(new_path):
        with open(new_path, "r", encoding="utf-8") as f:
            new_data = json.load(f)

    existing_ids = set()
    for d in existing:
        eid = d.get("id", "")
        if not eid and "conversations" in d:
            eid = str(hash(d["conversations"][0]["value"][:100]))
        existing_ids.add(eid)

    added = 0
    for d in new_data:
        did = d.get("id", "")
        if did not in existing_ids:
            # Convert to conversations format if needed
            if "conversations" not in d and "content" in d:
                entry = {
                    "id": did,
                    "question": d["question"],
                    "gt": d["gt"],
                    "content": d["content"],
                }
            else:
                entry = d
            existing.append(entry)
            existing_ids.add(did)
            added += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    return len(existing), added


def generate_predictions_for_distill(model_dir, raw_train_path, output_path,
                                     num_gpus=4, batch_size=16):
    """Use the trained model to generate predictions on raw training data for distillation."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    predict_script = os.path.join(script_dir, "..", "eval", "predict.py")

    cmd = (
        f"torchrun --nproc_per_node={num_gpus} {predict_script} "
        f"--mode cot "
        f"--model_dir {model_dir} "
        f"--test_path {raw_train_path} "
        f"--output_path {output_path} "
        f"--batch_size {batch_size}"
    )
    run_cmd(cmd, "Generating predictions on raw training data")


def filter_correct_predictions(predictions_path, raw_train_path, output_json_path):
    """Filter predictions that exactly match ground truth, format as distilled data."""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    # Load raw training data for ground truth
    gt_map = {}
    desc_map = {}
    with open(raw_train_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) >= 3:
                gt_map[row[0]] = row[1]
                desc_map[row[0]] = row[2]

    # Load predictions
    correct = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) < 3:
                continue
            cid, gt_smiles, pred_smiles = row[0], row[1], row[2]

            # Clean prediction
            for tok in ["<|im_end|>", "<|endoftext|>", "</s>", "[END_I_SMILES]"]:
                pred_smiles = pred_smiles.replace(tok, "")
            pred_smiles = pred_smiles.strip()

            # Check exact match via InChI
            try:
                m_gt = Chem.MolFromSmiles(gt_smiles)
                m_pred = Chem.MolFromSmiles(pred_smiles)
                if m_pred is not None and m_gt is not None:
                    if Chem.MolToInchi(m_pred) == Chem.MolToInchi(m_gt):
                        correct.append({
                            "id": cid,
                            "question": desc_map.get(cid, ""),
                            "gt": gt_smiles,
                            "content": f"<think>\n(model-generated reasoning)\n</think>\n<answer>\n{pred_smiles}\n</answer>",
                            "pred": pred_smiles,
                            "score": 1,
                        })
            except Exception:
                continue

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(correct, f, ensure_ascii=False, indent=2)

    return len(correct)


def main():
    parser = argparse.ArgumentParser(description="MoIA: Molecular Iterative Adaptation (SFT → RPO → Distill loop)")

    # Model
    parser.add_argument("--base_model", type=str, default="facebook/galactica-125m",
                        help="Base pretrained model name/path")

    # Data paths
    parser.add_argument("--initial_data", type=str, required=True,
                        help="Initial distilled training data (JSON)")
    parser.add_argument("--raw_train_data", type=str, required=True,
                        help="Raw training data (chebi-20 TSV format)")
    parser.add_argument("--test_data", type=str, default=None,
                        help="Test data for evaluation (chebi-20 TSV format)")

    # Output
    parser.add_argument("--work_dir", type=str, default="./iterative_train",
                        help="Working directory for all outputs")

    # Loop control
    parser.add_argument("--max_iterations", type=int, default=5,
                        help="Maximum number of SFT→GRPO→Distill iterations")
    parser.add_argument("--max_total_steps", type=int, default=100000,
                        help="Stop if total training steps exceed this")
    parser.add_argument("--min_new_data_ratio", type=float, default=0.01,
                        help="Stop if new data / total data < this ratio")

    # SFT hyperparams
    parser.add_argument("--sft_epochs", type=int, default=5)
    parser.add_argument("--sft_lr", type=float, default=1e-5)
    parser.add_argument("--sft_batch_size", type=int, default=1)
    parser.add_argument("--sft_grad_accum", type=int, default=4)
    parser.add_argument("--sft_max_seq_len", type=int, default=2048)

    # RPO hyperparams
    parser.add_argument("--rpo_epochs", type=int, default=2)
    parser.add_argument("--rpo_lr", type=float, default=1e-6)
    parser.add_argument("--rpo_num_generations", type=int, default=5)
    parser.add_argument("--rpo_temperature", type=float, default=1.0)
    parser.add_argument("--rpo_kl_coef", type=float, default=0.01)
    parser.add_argument("--rpo_global_batch_size", type=int, default=128)
    parser.add_argument("--rpo_rollout_batch_size", type=int, default=512)
    parser.add_argument("--rpo_micro_batch_size", type=int, default=4)

    # Distillation
    parser.add_argument("--distill_model", type=str, default="gemma3:12b",
                        help="Teacher model for rejection-sampling distillation")
    parser.add_argument("--distill_api_base", type=str, default="http://localhost:11434/v1/")
    parser.add_argument("--distill_score_threshold", type=int, default=8)
    parser.add_argument("--use_model_predictions", action="store_true", default=False,
                        help="Also use correct model predictions as new training data")

    # Hardware
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--predict_batch_size", type=int, default=16)

    args = parser.parse_args()

    # Setup directories
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).parent
    sft_script = script_dir / ".." / "train" / "sft_train.py"
    rpo_script = script_dir / ".." / "train" / "grpo_train.py"
    distill_script = script_dir / "rejection_sampling_distill.py"
    predict_script = script_dir / ".." / "eval" / "predict.py"
    eval_script = script_dir / ".." / "eval" / "evaluate.py"

    # Initialize training data
    current_data_path = str(work_dir / "train_data_iter0.json")
    if not os.path.exists(current_data_path):
        subprocess.run(["cp", args.initial_data, current_data_path], check=True)

    total_steps = 0
    log_entries = []

    for iteration in range(args.max_iterations):
        print(f"\n{'#'*60}")
        print(f"# ITERATION {iteration + 1} / {args.max_iterations}")
        print(f"{'#'*60}")

        iter_dir = work_dir / f"iter_{iteration}"
        iter_dir.mkdir(exist_ok=True)

        data_count = count_json_entries(current_data_path)
        print(f"Training data size: {data_count}")

        # ============================================================
        # STEP 1: SFT
        # ============================================================
        sft_output = str(iter_dir / "sft_output")
        sft_cmd = (
            f"torchrun --nproc_per_node={args.num_gpus} {sft_script} "
            f"--mode cot "
            f"--model_name {args.base_model} "
            f"--data_path {current_data_path} "
            f"--output_dir {sft_output} "
            f"--epochs {args.sft_epochs} "
            f"--max_seq_len {args.sft_max_seq_len} "
            f"--batch_size {args.sft_batch_size} "
            f"--grad_accum {args.sft_grad_accum} "
            f"--lr {args.sft_lr}"
        )
        run_cmd(sft_cmd, f"SFT training (iteration {iteration + 1})")

        sft_steps = (data_count // (args.sft_batch_size * args.sft_grad_accum * args.num_gpus) + 1) * args.sft_epochs
        total_steps += sft_steps

        # ============================================================
        # STEP 2: RPO (Reinforced Policy Optimization)
        # ============================================================
        sft_model_dir = f"{sft_output}/final"
        rpo_output = str(iter_dir / "rpo_output")
        rpo_cmd = (
            f"torchrun --nproc_per_node={args.num_gpus} {rpo_script} "
            f"--sft_model_dir {sft_model_dir} "
            f"--data_path {current_data_path} "
            f"--output_dir {rpo_output} "
            f"--epochs {args.rpo_epochs} "
            f"--lr {args.rpo_lr} "
            f"--num_generations {args.rpo_num_generations} "
            f"--temperature {args.rpo_temperature} "
            f"--beta {args.rpo_kl_coef} "
            f"--global_batch_size {args.rpo_global_batch_size} "
            f"--rollout_batch_size {args.rpo_rollout_batch_size} "
            f"--micro_batch_size {args.rpo_micro_batch_size}"
        )
        run_cmd(rpo_cmd, f"RPO training (iteration {iteration + 1})")

        rpo_steps = (data_count // (args.rpo_micro_batch_size * args.num_gpus) + 1) * args.rpo_epochs
        total_steps += rpo_steps

        # ============================================================
        # STEP 3: Evaluate (optional)
        # ============================================================
        rpo_model_dir = f"{rpo_output}/final"
        if args.test_data:
            pred_path = str(iter_dir / "predictions.txt")
            pred_cmd = (
                f"torchrun --nproc_per_node={args.num_gpus} {predict_script} "
                f"--mode cot "
                f"--model_dir {rpo_model_dir} "
                f"--test_path {args.test_data} "
                f"--output_path {pred_path} "
                f"--batch_size {args.predict_batch_size}"
            )
            run_cmd(pred_cmd, f"Evaluation prediction (iteration {iteration + 1})")

            eval_cmd = f"python {eval_script} {pred_path}"
            run_cmd(eval_cmd, f"Evaluation metrics (iteration {iteration + 1})")

        # ============================================================
        # STEP 4: Rejection-sampling distillation on raw training data
        # ============================================================
        new_distill_path = str(iter_dir / "new_distilled.json")
        distill_cmd = (
            f"python {distill_script} "
            f"--model {args.distill_model} "
            f"--api_base {args.distill_api_base} "
            f"--train_data {args.raw_train_data} "
            f"--output {new_distill_path} "
            f"--score_threshold {args.distill_score_threshold} "
            f"--existing_ids {current_data_path}"
        )
        run_cmd(distill_cmd, f"Rejection-sampling distillation (iteration {iteration + 1})")

        # Optionally also use correct model predictions
        model_pred_count = 0
        if args.use_model_predictions:
            train_pred_path = str(iter_dir / "train_predictions.txt")
            train_pred_cmd = (
                f"torchrun --nproc_per_node={args.num_gpus} {predict_script} "
                f"--mode cot "
                f"--model_dir {rpo_model_dir} "
                f"--test_path {args.raw_train_data} "
                f"--output_path {train_pred_path} "
                f"--batch_size {args.predict_batch_size}"
            )
            run_cmd(train_pred_cmd, "Generating predictions on raw train for self-distillation")

            model_correct_path = str(iter_dir / "model_correct.json")
            model_pred_count = filter_correct_predictions(
                train_pred_path, args.raw_train_data, model_correct_path
            )
            print(f"Model correct predictions: {model_pred_count}")

            # Merge model predictions into new distilled data
            if model_pred_count > 0:
                merge_distilled_data(new_distill_path, model_correct_path, new_distill_path)

        # ============================================================
        # STEP 5: Merge new data and check stopping criteria
        # ============================================================
        next_data_path = str(work_dir / f"train_data_iter{iteration + 1}.json")
        new_total, new_added = merge_distilled_data(
            current_data_path, new_distill_path, next_data_path
        )

        log_entry = {
            "iteration": iteration + 1,
            "prev_data_count": data_count,
            "new_distilled": count_json_entries(new_distill_path),
            "model_correct": model_pred_count,
            "new_added": new_added,
            "total_data": new_total,
            "total_steps": total_steps,
        }
        log_entries.append(log_entry)
        print(f"\n[SUMMARY] Iteration {iteration + 1}:")
        print(f"  Previous data: {data_count}")
        print(f"  New data added: {new_added}")
        print(f"  Total data: {new_total}")
        print(f"  Total training steps: {total_steps}")

        # Save log
        with open(str(work_dir / "training_log.json"), "w") as f:
            json.dump(log_entries, f, indent=2)

        # Check stopping criteria
        if new_added == 0:
            print("\n[STOP] No new data added. Training converged.")
            break

        if data_count > 0 and new_added / data_count < args.min_new_data_ratio:
            print(f"\n[STOP] New data ratio ({new_added}/{data_count} = "
                  f"{new_added/data_count:.4f}) below threshold ({args.min_new_data_ratio}).")
            break

        if total_steps >= args.max_total_steps:
            print(f"\n[STOP] Total steps ({total_steps}) reached max ({args.max_total_steps}).")
            break

        current_data_path = next_data_path

    print(f"\n{'='*60}")
    print(f"Training complete after {len(log_entries)} iterations.")
    print(f"Final data size: {log_entries[-1]['total_data']}")
    print(f"Total training steps: {total_steps}")
    print(f"Logs saved to {work_dir / 'training_log.json'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

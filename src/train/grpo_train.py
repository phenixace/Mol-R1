"""RPO (Reinforced Policy Optimization) training for molecular SMILES generation.

Builds on an SFT-trained model, using exact-match and validity rewards.
"""

import json
import os
import re
import torch

import transformers.modeling_utils as _mu
import transformers.pytorch_utils as _pu
import torch.distributed.tensor as _dt
if not hasattr(_dt, "DTensor"):
    class _FakeDTensor:
        pass
    _dt.DTensor = _FakeDTensor
    _mu.DTensor = _FakeDTensor
    if not hasattr(_pu, "DTensor"):
        _pu.DTensor = _FakeDTensor

from argparse import ArgumentParser
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig

_orig_get_train_sampler = GRPOTrainer._get_train_sampler
def _patched_get_train_sampler(self, dataset=None):
    return _orig_get_train_sampler(self)
GRPOTrainer._get_train_sampler = _patched_get_train_sampler

COT_PROMPT = (
    "Based on the description, generate the SMILES of the molecule with reasoning.\n"
    "Description: {question}\n\n<think>"
)
CONV_SUFFIX = " Please help me generate a molecule SMILES based on the above description."


def extract_smiles(text):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"</think>\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip().split("\n")[-1].strip()


def exact_match_reward(prompts, completions, gt, **kwargs):
    rewards = []
    for completion, ground_truth in zip(completions, gt):
        pred = extract_smiles(completion)
        rewards.append(1.0 if pred.strip() == ground_truth.strip() else 0.0)
    return rewards


def validity_reward(prompts, completions, **kwargs):
    from rdkit import Chem
    rewards = []
    for completion in completions:
        pred = extract_smiles(completion)
        mol = Chem.MolFromSmiles(pred.strip())
        rewards.append(0.5 if mol is not None else 0.0)
    return rewards


def parse_entry(d):
    if "question" in d:
        return d["question"], d.get("gt", "")
    convs = d["conversations"]
    question = convs[0]["value"]
    if question.endswith(CONV_SUFFIX):
        question = question[: -len(CONV_SUFFIX)]
    gpt_text = convs[1]["value"]
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", gpt_text, re.DOTALL)
    gt = m.group(1).strip() if m else ""
    return question, gt


def main():
    parser = ArgumentParser()
    parser.add_argument("--sft_model_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="grpo_output")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--num_generations", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_completion_length", type=int, default=10000)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--global_batch_size", type=int, default=128)
    parser.add_argument("--rollout_batch_size", type=int, default=512)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    args = parser.parse_args()

    with open(args.data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    parsed = [parse_entry(d) for d in raw_data]
    dataset = Dataset.from_dict({
        "prompt": [COT_PROMPT.format(question=q) for q, gt in parsed],
        "gt": [gt for q, gt in parsed],
    })

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_dir)
    is_galactica = "galactica" in args.sft_model_dir.lower()
    if is_galactica:
        tokenizer.pad_token_id = 1
        tokenizer.eos_token_id = 2
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=max(1, args.global_batch_size // (args.micro_batch_size * 8)),
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        max_prompt_length=512,
        beta=args.beta,
        reward_weights=[1.0, 0.5],
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_dir, torch_dtype=torch.bfloat16
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[exact_match_reward, validity_reward],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    save_dir = os.path.join(args.output_dir, "final")
    if trainer.is_world_process_zero():
        os.makedirs(save_dir, exist_ok=True)
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        from safetensors.torch import save_file

        state_dict = {
            k: v.cpu().contiguous() for k, v in unwrapped.state_dict().items()
        }
        save_file(state_dict, os.path.join(save_dir, "model.safetensors"))
        unwrapped.config.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"RPO training complete. Model saved to {save_dir}")


if __name__ == "__main__":
    main()

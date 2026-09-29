"""Explore operator: GRPO on top of the SFT policy.

Prompts come from the raw training split (TSV) or a trace set (JSON), in the chat
format of src.common.chat_prompt. The reward is the verifier shared with the harvest
step: 1 if the completion is <think>...</think><answer>...</answer> and the answer is
the reference molecule (InChI identity), else 0. A SMILES-validity reward can be
added with --validity_weight (off by default).
"""

import os
import sys
from pathlib import Path

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import chat_prompt, extract_answer, is_valid_smiles, load_items, verify

_orig_get_train_sampler = GRPOTrainer._get_train_sampler
def _patched_get_train_sampler(self, dataset=None):
    return _orig_get_train_sampler(self)
GRPOTrainer._get_train_sampler = _patched_get_train_sampler

def exact_match_reward(prompts, completions, gt, **kwargs):
    return [1.0 if verify(c, g)[0] else 0.0 for c, g in zip(completions, gt)]


def validity_reward(prompts, completions, **kwargs):
    return [0.5 if is_valid_smiles(extract_answer(c)) else 0.0 for c in completions]


def main():
    parser = ArgumentParser()
    parser.add_argument("--sft_model_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True,
                        help="Raw training split (TSV) or trace set (JSON)")
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
    parser.add_argument("--em_weight", type=float, default=1.0)
    parser.add_argument("--validity_weight", type=float, default=0.0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    items = load_items(args.data_path)
    dataset = Dataset.from_dict({
        "prompt": [chat_prompt(tokenizer, it["question"]) for it in items],
        "gt": [it["gt"] for it in items],
    })

    rewards = [(f, w) for f, w in ((exact_match_reward, args.em_weight),
                                   (validity_reward, args.validity_weight)) if w > 0]

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=max(
            1, args.global_batch_size // (args.micro_batch_size * int(os.environ.get("WORLD_SIZE", 1)))),
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
        reward_weights=[w for _, w in rewards],
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_dir, torch_dtype=torch.bfloat16
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[f for f, _ in rewards],
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

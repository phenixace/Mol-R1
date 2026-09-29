"""SFT (Supervised Fine-Tuning) for molecular SMILES generation.

Supports two modes:
  - cot: Chain-of-Thought reasoning then answer
  - gt:  Direct ground-truth SMILES generation
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# DTensor compatibility patch for older PyTorch versions
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

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import COT_PREFIX as COT_PROMPT, GT_PROMPT, parse_entry

DEFAULT_MODEL_NAME = "facebook/galactica-125m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cot", "gt"], required=True)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"sft_output_{args.mode}"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    is_galactica = "galactica" in args.model_name.lower()
    if is_galactica:
        tokenizer.pad_token_id = 1
        tokenizer.eos_token_id = 2
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    with open(args.data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    parsed = [parse_entry(d) for d in raw_data]
    if args.mode == "cot":
        texts = [COT_PROMPT.format(question=q) + content for q, gt, content in parsed]
        response_template = "\n\n<think>"
    else:
        texts = [
            GT_PROMPT.format(question=q) + gt + "[END_I_SMILES]"
            for q, gt, content in parsed
        ]
        response_template = "[START_I_SMILES]"

    dataset = Dataset.from_dict({"text": texts})

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        report_to="none",
        gradient_checkpointing=True,
        max_seq_length=args.max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
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
        print(f"SFT training complete ({args.mode}). Model saved to {save_dir}")


if __name__ == "__main__":
    main()

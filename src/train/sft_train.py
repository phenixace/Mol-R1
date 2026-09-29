"""Imitate operator: supervised fine-tuning on a trace set.

Each example is one chat exchange in the format of src.common.chat_messages; the loss
covers the response and its end-of-turn token, not the prompt. Two modes:
  - cot: the response is the reasoning trace <think>...</think><answer>...</answer>
  - gt:  the response is <answer>SMILES</answer> (direct supervision, no reasoning)
"""

import argparse
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
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                          TrainingArguments)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import chat_example, load_json, parse_entry

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def tokenize(tokenizer, question, response, max_seq_len):
    """Input ids of prompt + response, with the prompt masked out of the labels."""
    prompt, target = chat_example(tokenizer, question, response)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + target_ids)[:max_seq_len]
    labels = ([-100] * len(prompt_ids) + target_ids)[:max_seq_len]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    examples = []
    for question, gt, content in map(parse_entry, load_json(args.data_path)):
        response = content if args.mode == "cot" else f"<answer>\n{gt}\n</answer>"
        examples.append(tokenize(tokenizer, question, response, args.max_seq_len))
    dataset = Dataset.from_list(examples)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    training_args = TrainingArguments(
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
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

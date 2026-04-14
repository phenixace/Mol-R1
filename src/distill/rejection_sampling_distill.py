"""Rejection-sampling distillation: generate reasoning traces from a teacher model,
score them, and keep only high-quality samples (score >= threshold).

This script uses an OpenAI-compatible API (e.g. Ollama, vLLM) to:
1. Generate <think>...</think><answer>...</answer> reasoning traces
2. Score each trace on correctness, knowledge richness, and logic coherence
3. Keep traces scoring >= threshold; re-generate low-scoring ones (up to max_retries)
"""

import argparse
import csv
import json
import os
import re
import openai
from tqdm import tqdm


CONTEXT_EXAMPLE = """Caption: The molecule is a cyclodiene organochlorine insecticide. It has a role as a GABA-gated chloride channel antagonist and a persistent organic pollutant. It derives from a hydride of an indene.

Molecule: C1C2C(C(C1Cl)Cl)C3(C(=C(C2(C3(Cl)Cl)Cl)Cl)Cl)Cl

Reasoning Process from Caption to Molecule:
<think>
Okay, I'm now tasked with generating the SMILES representation of a molecule based on the user's description. First, I need to carefully analyze each sentence and understand the corresponding structural features.
The first sentence: "The molecule is a cyclodiene organochlorine insecticide." There are several keywords here: cyclodiene, organochlorine, and insecticide. Let me think, what does a cyclodiene SMILES look like?
There seem to be several types of cyclodienes, differentiated by the number of carbon atoms, such as cyclopentadiene (C1=CC=CC1) and cyclohexadiene (C1=CCC=CC1, C1C=CC=CC1). However, the description only mentions cyclodiene, and I'm currently unsure which one it is.
Well, I can look for information elsewhere. According to the second keyword, this molecule is an organochlorine compound, right? So, there must be chlorine atoms in the molecule's SMILES!
The second sentence: "It has a role as a GABA-gated chloride channel antagonist and a persistent organic pollutant." This part mainly describes the function and properties of the molecule. A GABA-gated chloride channel antagonist indicates that this is a type of drug that affects the central nervous system.
The third sentence: "It derives from a hydride of an indene." Here, indene is mentioned, and the molecule I need to generate is its derivative. Indene is a bicyclic structure consisting of a benzene ring and a cyclopentene.
Based on the structure of common chlorinated insecticides, I find that these compounds have multiple chlorine atoms, and their chemical properties require the presence of double bonds. Therefore, the final SMILES representation of this molecule is: C1C2C(C(C1Cl)Cl)C3(C(=C(C2(C3(Cl)Cl)Cl)Cl)Cl)Cl.
</think>
<answer>
C1C2C(C(C1Cl)Cl)C3(C(=C(C2(C3(Cl)Cl)Cl)Cl)Cl)Cl
</answer>
"""


def create_distill_prompt(caption, molecule):
    return (
        CONTEXT_EXAMPLE
        + "\nNow, please write out the reasoning process from Caption to Molecule "
        "based on the following pair of Caption-Molecule. Remember, you need to imitate "
        "the above example for reasoning, rather than assuming you already know the answer "
        "from the beginning. In addition, you should pay attention to not skipping some "
        "important reasoning steps and processes.\n\n"
        f"Caption: {caption}\n\nMolecule: {molecule}\n\n"
        "Reasoning Process from Caption to Molecule:\n"
        "Your output should strictly follow the format of <think> ... </think> and "
        "<answer> ... </answer>. Please ensure that the reasoning process is clear "
        "and the final answer is correct."
    )


def create_score_prompt(caption, molecule, reasoning):
    return (
        "Now, please provide your evaluation of the reasoning process from Caption "
        "to Molecule as follows. Your score must ensure the correctness of the reasoning "
        "process, the richness of knowledge, and the coherence of logic, and finally "
        "output the total score by adding them up. Additionally, you need to ensure that "
        "the reasoning process does not directly reveal molecular data (if violated, "
        "the total score is 0 points).\n\n"
        f"Caption: {caption}\n\nMolecule: {molecule}\n\n"
        f"Reasoning Process from Caption to Molecule: {reasoning}\n\n"
        r"Now, please provide your total score (ranging from 0 to 10) and output it "
        r"in the form of \boxed{}."
    )


def distill_and_score(client, model, caption, molecule, score_threshold=8, max_retries=5):
    """Generate a reasoning trace and score it. Returns (trace, score) or (None, -1)."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an excellent chemist."},
                    {"role": "user", "content": create_distill_prompt(caption, molecule)},
                ],
                model=model,
                timeout=60,
            )
            raw = response.choices[0].message.content

            if not all(tag in raw for tag in ["<think>", "</think>", "<answer>", "</answer>"]):
                continue

            # Score the reasoning
            for score_attempt in range(3):
                try:
                    score_resp = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are an excellent chemist."},
                            {"role": "user", "content": create_score_prompt(caption, molecule, raw)},
                        ],
                        model=model,
                        timeout=30,
                    )
                    score_raw = score_resp.choices[0].message.content
                    score_match = re.search(r"\\boxed{(\d+)}", score_raw)
                    if not score_match:
                        score_match = re.search(r"boxed\{(\d+)\}", score_raw)
                    if not score_match:
                        continue

                    score = int(score_match.group(1))
                    if 0 <= score <= 10 and score >= score_threshold:
                        reason = raw.split("<think>")[1].split("</think>")[0].strip()
                        answer = raw.split("<answer>")[1].split("</answer>")[0].strip()
                        clean_trace = f"<think>\n{reason}\n</think>\n<answer>\n{answer}\n</answer>"
                        return clean_trace, score
                    break
                except Exception:
                    continue
        except Exception:
            continue

    return None, -1


def load_raw_train_data(path):
    """Load chebi-20 format TSV: CID, SMILES, description."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) >= 3:
                data.append({
                    "id": row[0],
                    "molecule": row[1],
                    "instruction": row[2],
                })
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma3:12b")
    parser.add_argument("--api_base", type=str, default="http://localhost:11434/v1/")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to raw training data (chebi-20 TSV format)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file for distilled data")
    parser.add_argument("--score_threshold", type=int, default=8)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--existing_ids", type=str, default=None,
                        help="Optional JSON file with already-distilled IDs to skip")
    args = parser.parse_args()

    client = openai.OpenAI(base_url=args.api_base, api_key="ollama")

    data = load_raw_train_data(args.train_data)
    print(f"Loaded {len(data)} training samples")

    # Load existing results if resuming
    distilled = []
    existing_ids = set()
    if args.output and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            distilled = json.load(f)
        existing_ids = {d["id"] for d in distilled}
        print(f"Resuming: {len(existing_ids)} already distilled")

    if args.existing_ids and os.path.exists(args.existing_ids):
        with open(args.existing_ids, "r", encoding="utf-8") as f:
            extra = json.load(f)
        for d in extra:
            existing_ids.add(d.get("id", ""))

    for item in tqdm(data, desc="Distilling"):
        if item["id"] in existing_ids:
            continue

        trace, score = distill_and_score(
            client, args.model, item["instruction"], item["molecule"],
            score_threshold=args.score_threshold, max_retries=args.max_retries,
        )

        if trace is not None:
            distilled.append({
                "id": item["id"],
                "question": item["instruction"],
                "gt": item["molecule"],
                "content": trace,
                "score": score,
            })
            existing_ids.add(item["id"])

            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(distilled, f, ensure_ascii=False, indent=2)

    print(f"Distillation complete: {len(distilled)} samples saved to {args.output}")


if __name__ == "__main__":
    main()

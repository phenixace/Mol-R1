"""PRID (Prior Regulation via In-context Distillation): generate reasoning traces
guided by in-context examples with known caption-molecule pairs.

Unlike rejection sampling which generates and filters, PRID provides the ground-truth
molecule to the teacher model and asks it to write the reasoning process, guided by
a detailed in-context example that demonstrates the expected reasoning style.

Supports OpenAI-compatible APIs (GPT-4o, Ollama, vLLM, etc.).
"""

import argparse
import csv
import json
import os
import openai
from tqdm import tqdm


ICL_EXAMPLE = """Caption: The molecule is a cyclodiene organochlorine insecticide. It has a role as a GABA-gated chloride channel antagonist and a persistent organic pollutant. It derives from a hydride of an indene.

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

USER_PROMPT_TEMPLATE = (
    "Now, please write out the reasoning process from Caption to Molecule "
    "based on the following pair of Caption-Molecule. Remember, you need to imitate "
    "the above example for reasoning, rather than assuming you already know the answer "
    "from the beginning. In addition, you should pay attention to not skipping some "
    "important reasoning steps and processes.\n\n"
    "Caption: {caption}\n\nMolecule: {molecule}\n\n"
    "Reasoning Process from Caption to Molecule:\n"
    "Your output should strictly follow the format of <think> ... </think> and "
    "<answer> ... </answer>. Please ensure that the reasoning process is clear "
    "and the final answer is correct."
)


def distill_one(client, model, caption, molecule, max_retries=3, temperature=0.6, max_tokens=4096):
    """Generate a PRID reasoning trace for one caption-molecule pair."""
    system_msg = "You are an excellent chemist who is familiar with the SMILES representation of molecules."
    user_prompt = ICL_EXAMPLE + "\n\n" + USER_PROMPT_TEMPLATE.format(caption=caption, molecule=molecule)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=120,
            )
            raw = response.choices[0].message.content

            if "<think>" in raw and "</think>" in raw and "<answer>" in raw and "</answer>" in raw:
                reason = raw.split("<think>")[1].split("</think>")[0].strip()
                answer = raw.split("<answer>")[1].split("</answer>")[0].strip()
                clean_trace = f"<think>\n{reason}\n</think>\n<answer>\n{answer}\n</answer>"
                return clean_trace
            return raw
        except Exception as e:
            print(f"[Retry {attempt+1}/{max_retries}] {e}")
            continue

    return None


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
    parser = argparse.ArgumentParser(description="PRID: Prior Regulation via In-context Distillation")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Teacher model name (gpt-4o, gemma3:12b, etc.)")
    parser.add_argument("--api_base", type=str, default=None,
                        help="OpenAI-compatible API base URL (e.g. http://localhost:11434/v1/ for Ollama)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (use 'ollama' for Ollama)")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to raw training data (chebi-20 TSV format)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file for distilled data")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()

    client_kwargs = {}
    if args.api_base:
        client_kwargs["base_url"] = args.api_base
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    else:
        client_kwargs["api_key"] = os.environ.get("OPENAI_API_KEY", "ollama")

    client = openai.OpenAI(**client_kwargs)

    data = load_raw_train_data(args.train_data)
    print(f"Loaded {len(data)} training samples")

    distilled = []
    existing_ids = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            distilled = json.load(f)
        existing_ids = {d["id"] for d in distilled}
        print(f"Resuming: {len(existing_ids)} already distilled")

    for item in tqdm(data, desc="PRID distilling"):
        if item["id"] in existing_ids:
            continue

        trace = distill_one(
            client, args.model, item["instruction"], item["molecule"],
            max_retries=args.max_retries,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        if trace is not None:
            distilled.append({
                "id": item["id"],
                "question": item["instruction"],
                "gt": item["molecule"],
                "content": trace,
            })
            existing_ids.add(item["id"])

            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(distilled, f, ensure_ascii=False, indent=2)

    print(f"PRID distillation complete: {len(distilled)} samples saved to {args.output}")


if __name__ == "__main__":
    main()

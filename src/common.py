"""Shared prompt format, answer extraction and the answer verifier.

Every stage prompts the model the same way: one user turn holding the description
followed by INSTRUCTION, rendered with the model's chat template and no system
message. This is the format of the conversation-format seed (T0.json). The model
answers <think>...</think><answer>SMILES</answer>.

The same verifier (InChI-level molecule identity) is used by the RL reward, the
harvest step of the self-improvement loop, and evaluation, so the policy is
optimized, harvested and scored against one definition of "correct".
"""

import csv
import json
import re
from functools import lru_cache

# Appended to the description to form the user turn.
INSTRUCTION = " Please help me generate a molecule SMILES based on the above description."

STRIP_TOKENS = ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "</s>"]

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_WELL_FORMED = re.compile(r"^\s*<think>.*?</think>\s*<answer>(.*?)</answer>\s*$", re.DOTALL)


def chat_messages(question, response=None):
    """The user turn for a description, followed by the response if one is given."""
    messages = [{"role": "user", "content": question + INSTRUCTION}]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return messages


def chat_prompt(tokenizer, question):
    """Generation prompt: the chat template applied to the user turn, up to the assistant header."""
    return tokenizer.apply_chat_template(chat_messages(question), tokenize=False,
                                         add_generation_prompt=True)


def chat_example(tokenizer, question, response):
    """(prompt, target) texts of an SFT example; the prompt is exactly chat_prompt()."""
    prompt = chat_prompt(tokenizer, question)
    full = tokenizer.apply_chat_template(chat_messages(question, response), tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("The chat template does not render the generation prompt as a prefix "
                         "of the conversation.")
    return prompt, full[len(prompt):]


def clean_smiles(s):
    for tok in STRIP_TOKENS:
        s = s.replace(tok, "")
    return s.strip()


def extract_answer(text):
    """The prediction: the last <answer>...</answer> block with whitespace removed, or ""."""
    blocks = _ANSWER.findall(text)
    return "".join(clean_smiles(blocks[-1]).split()) if blocks else ""


def well_formed_answer(content):
    """The answer if content is exactly <think>...</think><answer>...</answer>, else None."""
    m = _WELL_FORMED.match(clean_smiles(content))
    return "".join(m.group(1).split()) if m else None


_rdkit_quiet = False


def _rdkit():
    global _rdkit_quiet
    from rdkit import Chem, RDLogger
    if not _rdkit_quiet:
        RDLogger.DisableLog("rdApp.*")
        _rdkit_quiet = True
    return Chem


@lru_cache(maxsize=None)
def to_inchi(smiles):
    """InChI of a SMILES string, or None if it does not parse."""
    s = clean_smiles(smiles)
    if not s:
        return None
    Chem = _rdkit()
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    try:
        inchi = Chem.MolToInchi(mol)
    except Exception:
        return None
    return inchi or None


def is_valid_smiles(smiles):
    s = clean_smiles(smiles)
    return bool(s) and _rdkit().MolFromSmiles(s) is not None


def same_molecule(pred, gt):
    """True iff both parse and have identical InChI (the exact-match criterion)."""
    gi = to_inchi(gt)
    return gi is not None and to_inchi(pred) == gi


def verify(content, gt, require_format=True):
    """Verify a reasoning trace against the reference molecule.

    Returns (accepted, predicted_smiles). With require_format, the trace must be
    <think>...</think><answer>...</answer> and the answer is read from the tag.
    """
    pred = well_formed_answer(content) if require_format else extract_answer(content)
    if not pred:
        return False, None
    return same_molecule(pred, gt), pred


def parse_entry(d):
    """Return (question, gt, content) from a flat or conversation-format entry."""
    if "conversations" not in d:
        return d.get("question", ""), d.get("gt", ""), d.get("content", "")
    convs = d["conversations"]
    question = convs[0]["value"]
    if question.endswith(INSTRUCTION.lstrip()):
        question = question[: -len(INSTRUCTION.lstrip())].rstrip()
    content = convs[1]["value"]
    return question, extract_answer(content), content


def load_tsv(path):
    """Load a ChEBI-20 split (CID \\t SMILES \\t description) as items."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) >= 3:
                items.append({"id": row[0], "gt": row[1], "question": row[2]})
    return items


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path, indent=2):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_items(path):
    """Load prompts with references from a TSV split or a JSON trace set."""
    if path.endswith((".txt", ".tsv")):
        return load_tsv(path)
    items = []
    for d in load_json(path):
        question, gt, _ = parse_entry(d)
        items.append({"id": d.get("id"), "gt": gt, "question": question})
    return items

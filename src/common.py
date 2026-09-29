"""Shared prompts, parsing and the answer verifier.

The same verifier (InChI-level molecule identity) is used by the RL reward, the
harvest step of the self-improvement loop, and evaluation, so the policy is
optimized, harvested and scored against one definition of "correct".
"""

import csv
import json
import re
from functools import lru_cache

COT_PREFIX = (
    "Based on the description, generate the SMILES of the molecule with reasoning.\n"
    "Description: {question}\n\n"
)
# Generation prompt: the model continues after the opening <think> tag.
# Training text is COT_PREFIX + content, where content starts with "<think>".
GEN_PROMPT = COT_PREFIX + "<think>"
GT_PROMPT = (
    "Based on the description, generate the SMILES of the molecule.\n"
    "Description: {question}\n\n[START_I_SMILES]"
)
# Suffix appended to questions in the conversation-format seed file (T0_zh.json).
CONV_SUFFIX = " Please help me generate a molecule SMILES based on the above description."

STRIP_TOKENS = ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>", "[END_I_SMILES]"]

_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_AFTER_THINK = re.compile(r"</think>\s*(.*)", re.DOTALL)
_WELL_FORMED = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>\s*(.*?)\s*</answer>\s*$", re.DOTALL
)


def clean_smiles(s):
    for tok in STRIP_TOKENS:
        s = s.replace(tok, "")
    return s.strip()


def extract_answer(text):
    """Best-effort extraction of the final SMILES from a generation."""
    m = _ANSWER.search(text)
    if m:
        return clean_smiles(m.group(1))
    m = _AFTER_THINK.search(text)
    if m:
        return clean_smiles(m.group(1))
    lines = text.strip().split("\n")
    return clean_smiles(lines[-1]) if lines else ""


def well_formed_answer(content):
    """Return the answer if content is exactly <think>...</think><answer>...</answer>, else None."""
    m = _WELL_FORMED.match(clean_smiles(content))
    return clean_smiles(m.group(1)) if m else None


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
    if pred is None:
        return False, None
    return same_molecule(pred, gt), pred


def parse_entry(d):
    """Return (question, gt, content) from a flat or conversation-format entry."""
    if "conversations" not in d:
        return d.get("question", ""), d.get("gt", ""), d.get("content", "")
    convs = d["conversations"]
    question = convs[0]["value"]
    if question.endswith(CONV_SUFFIX):
        question = question[: -len(CONV_SUFFIX)]
    content = convs[1]["value"]
    m = _ANSWER.search(content)
    gt = m.group(1).strip() if m else ""
    return question, gt, content


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

"""Evaluate SMILES predictions against ground truth using standard molecular translation metrics."""

import sys
import csv
import numpy as np
from Levenshtein import distance as lev
from nltk.translate.bleu_score import corpus_bleu
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit import DataStructs, RDLogger

RDLogger.DisableLog('rdApp.*')


STRIP_TOKENS = ["<|im_end|>", "<|endoftext|>", "</s>", "[END_I_SMILES]"]


def clean_smiles(s):
    for tok in STRIP_TOKENS:
        s = s.replace(tok, "")
    return s.strip()


def load_predictions(path):
    cids, gts, preds = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) >= 3:
                cids.append(row[0])
                gts.append(row[1])
                preds.append(clean_smiles(row[2]))
    return cids, gts, preds


def evaluate(gts, preds, morgan_r=2):
    n = len(gts)
    assert n == len(preds)

    # --- BLEU (character-level) ---
    references = [[list(gt)] for gt in gts]
    hypotheses = [list(pred) for pred in preds]
    bleu = corpus_bleu(references, hypotheses)

    # --- Exact match (InChI-based), Levenshtein, Validity ---
    exact = 0
    levs = []
    bad_mols = 0
    valid_gt_mols, valid_pred_mols = [], []

    for gt, pred in zip(gts, preds):
        levs.append(lev(pred, gt))
        try:
            m_gt = Chem.MolFromSmiles(gt)
            m_pred = Chem.MolFromSmiles(pred)
            if m_pred is None:
                raise ValueError("Invalid prediction SMILES")
            if Chem.MolToInchi(m_pred) == Chem.MolToInchi(m_gt):
                exact += 1
            valid_gt_mols.append(m_gt)
            valid_pred_mols.append(m_pred)
        except:
            bad_mols += 1

    exact_match = exact / n
    levenshtein = np.mean(levs)
    validity = 1 - bad_mols / n

    # --- Fingerprint similarities (only for valid pairs) ---
    maccs_sims, rdk_sims, morgan_sims = [], [], []
    for m_gt, m_pred in zip(valid_gt_mols, valid_pred_mols):
        maccs_sims.append(DataStructs.FingerprintSimilarity(
            MACCSkeys.GenMACCSKeys(m_gt), MACCSkeys.GenMACCSKeys(m_pred),
            metric=DataStructs.TanimotoSimilarity))
        rdk_sims.append(DataStructs.FingerprintSimilarity(
            Chem.RDKFingerprint(m_gt), Chem.RDKFingerprint(m_pred),
            metric=DataStructs.TanimotoSimilarity))
        morgan_sims.append(DataStructs.TanimotoSimilarity(
            AllChem.GetMorganFingerprint(m_gt, morgan_r),
            AllChem.GetMorganFingerprint(m_pred, morgan_r)))

    maccs = np.mean(maccs_sims) if maccs_sims else 0
    rdk = np.mean(rdk_sims) if rdk_sims else 0
    morgan = np.mean(morgan_sims) if morgan_sims else 0

    return {
        "Total": n,
        "Exact Match": exact_match,
        "BLEU": bleu,
        "Levenshtein": levenshtein,
        "Validity": validity,
        "MACCS FTS": maccs,
        "RDK FTS": rdk,
        "Morgan FTS": morgan,
        "Valid Pairs": len(valid_gt_mols),
        "Invalid Predictions": bad_mols,
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "predictions_gt.txt"
    print(f"Evaluating: {path}")
    print("=" * 50)

    cids, gts, preds = load_predictions(path)
    results = evaluate(gts, preds)

    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()

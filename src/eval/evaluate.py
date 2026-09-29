"""Evaluate SMILES predictions against ground truth using standard molecular translation metrics."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from Levenshtein import distance as lev
from nltk.translate.bleu_score import corpus_bleu
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit import DataStructs, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common import clean_smiles, same_molecule

RDLogger.DisableLog('rdApp.*')


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
            if m_pred is None or m_gt is None:
                raise ValueError("Invalid SMILES")
            if same_molecule(pred, gt):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="predictions_gt.txt")
    parser.add_argument("--json_out", default=None, help="Also write metrics as JSON")
    args = parser.parse_args()
    print(f"Evaluating: {args.path}")
    print("=" * 50)

    cids, gts, preds = load_predictions(args.path)
    results = evaluate(gts, preds)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({k: float(v) for k, v in results.items()}, f, indent=2)

    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()

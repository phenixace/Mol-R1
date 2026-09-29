"""Unit tests for the loop logic that does not need GPUs.

    python -m unittest discover -s tests -v
"""

import collections
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import GEN_PROMPT, dump_json, load_json, verify
from src.rsi.harvest import harvest, merge_shards, shard_paths
from src.rsi.loop import merge_previous
from src.rsi.stats import compare, resolve_ids, summarize


class FakeSampler:
    """Completes item q correctly on its plan[q]-th attempt (0-based), wrongly otherwise."""

    def __init__(self, plan, answers):
        self.plan, self.answers = plan, answers
        self.attempts = collections.Counter()
        self.calls = []

    def __call__(self, prompts, n):
        self.calls.append((len(prompts), n))
        out = []
        for p in prompts:
            q = p.split("Description: ")[1].split("\n\n")[0]
            comps = []
            for _ in range(n):
                k = self.attempts[q]
                self.attempts[q] += 1
                smiles = self.answers[q] if self.plan[q] == k else "C"
                comps.append(f"\nstep {k}\n</think>\n<answer>\n{smiles}\n</answer>")
            out.append(comps)
        return out


class TestVerify(unittest.TestCase):
    def test_equivalent_smiles_accepted(self):
        self.assertEqual(verify("<think>x</think>\n<answer>\nOCC\n</answer>", "CCO"), (True, "OCC"))

    def test_wrong_molecule_rejected(self):
        self.assertEqual(verify("<think>x</think><answer>C</answer>", "CCO"), (False, "C"))

    def test_format_required(self):
        malformed = "<think>x<answer>CCO</answer>"
        self.assertEqual(verify(malformed, "CCO", require_format=True), (False, None))
        self.assertEqual(verify(malformed, "CCO", require_format=False), (True, "CCO"))

    def test_unparsable_reference_never_matches(self):
        self.assertFalse(verify("<think>x</think><answer>C1CC</answer>", "C1CC")[0])


class TestHarvest(unittest.TestCase):
    def test_rounds_attempts_and_failure_times(self):
        items = [
            {"id": "a", "question": "qa", "gt": "CCO"},        # right on attempt 0
            {"id": "b", "question": "qb", "gt": "c1ccccc1"},   # right on attempt 10 (round 2)
            {"id": "c", "question": "qc", "gt": "CC(=O)O"},    # never right
        ]
        sampler = FakeSampler({"qa": 0, "qb": 10, "qc": None},
                              {"qa": "OCC", "qb": "c1ccccc1", "qc": "CC(=O)O"})
        entries, meta = harvest(items, sampler, max_attempts=20, samples_per_round=8, log=lambda *_: None)

        self.assertEqual([e["id"] for e in entries], ["a", "b"])
        self.assertEqual([e["failure_times"] for e in entries], [0, 10])
        self.assertEqual(entries[0]["pred"], "OCC")
        self.assertTrue(entries[1]["content"].startswith("<think>\nstep 10"))
        self.assertTrue(entries[1]["content"].endswith("</answer>"))
        self.assertEqual(sampler.calls, [(3, 8), (2, 8), (1, 4)])  # last round capped at max_attempts
        self.assertEqual(meta["samples_generated"], 3 * 8 + 2 * 8 + 1 * 4)
        self.assertEqual((meta["accepted"], meta["unresolved"], meta["rounds"]), (2, 1, 3))

    def test_prompt_matches_training_format(self):
        seen = []
        harvest([{"id": "a", "question": "q", "gt": "C"}],
                lambda prompts, n: seen.extend(prompts) or [["</think><answer>C</answer>"] * n],
                max_attempts=1, samples_per_round=1, log=lambda *_: None)
        self.assertEqual(seen, [GEN_PROMPT.format(question="q")])
        self.assertTrue(seen[0].endswith("\n\n<think>"))

    def test_merge_shards_keeps_training_order(self):
        items = [{"id": i} for i in ("x", "y", "z")]
        with tempfile.TemporaryDirectory() as d:
            for shard, ids in ((0, ["z"]), (1, ["x"])):
                entries_path, meta_path = shard_paths(d, shard)
                dump_json([{"id": i} for i in ids], entries_path)
                dump_json({"items": 2, "accepted": 1, "unresolved": 1, "samples_generated": 10,
                           "max_attempts": 64, "samples_per_round": 8, "require_format": True}, meta_path)
            out = os.path.join(d, "R.json")
            meta = merge_shards(d, 2, items, out)
            self.assertEqual([e["id"] for e in load_json(out)], ["x", "z"])
            self.assertEqual((meta["items"], meta["accepted"], meta["samples_generated"]), (4, 2, 20))


class TestStats(unittest.TestCase):
    def test_compare(self):
        self.assertEqual(compare({"1", "2", "3"}, {"2", "3", "4", "5"}),
                         {"new": 2, "lost": 1, "retained": 2, "net_growth": 1 / 3})

    def test_resolve_ids_disambiguates_by_molecule(self):
        train = [{"id": "10", "question": "same text", "gt": "CCO"},
                 {"id": "11", "question": "same text", "gt": "CCN"},
                 {"id": "12", "question": "unique", "gt": "C"}]
        conv = lambda q, smi: {"conversations": [{"from": "human", "value": q},
                                                 {"from": "gpt", "value": f"<think></think><answer>{smi}</answer>"}]}
        entries = [conv("same text", "NCC"), conv("unique", "C"), {"id": 99}, conv("missing", "C")]
        self.assertEqual(resolve_ids(entries, train), ["11", "12", "99", None])

    def test_summarize_counts_duplicates(self):
        entries = [{"id": "1", "content": "ab", "failure_times": 0},
                   {"id": "1", "content": "abcd", "failure_times": 2},
                   {"id": "2", "content": "abc", "failure_times": 0}]
        summary, unique = summarize(entries, ["1", "1", "2"], universe_size=10)
        self.assertEqual(unique, {"1", "2"})
        self.assertEqual((summary["rows"], summary["unique_ids"], summary["duplicate_rows"]), (3, 2, 1))
        self.assertAlmostEqual(summary["coverage"], 0.2)
        self.assertAlmostEqual(summary["first_try_rate"], 2 / 3)


class TestMerge(unittest.TestCase):
    def test_new_trace_replaces_old_and_unsolved_are_carried_over(self):
        prev = [{"id": "1", "question": "q1", "gt": "C", "content": "t1", "failure_times": 3},
                {"id": "2", "question": "q2", "gt": "N", "content": "t2", "failure_times": 0}]
        harvested = [{"id": "1", "question": "q1", "gt": "C", "content": "new", "failure_times": 0}]
        merged = merge_previous(prev, ["1", "2"], harvested)
        self.assertEqual([(e["id"], e["content"]) for e in merged], [("1", "new"), ("2", "t2")])
        self.assertTrue(merged[1]["from_previous"])
        self.assertNotIn("from_previous", merged[0])

    def test_seed_entries_are_converted_to_flat_format(self):
        seed = [{"conversations": [{"from": "human", "value": "q"},
                                   {"from": "gpt", "value": "<think>r</think><answer>CCO</answer>"}]}]
        merged = merge_previous(seed, ["7"], [])
        self.assertEqual(merged, [{"question": "q", "gt": "CCO", "pred": "CCO", "score": 1,
                                   "content": "<think>r</think><answer>CCO</answer>",
                                   "id": "7", "from_previous": True}])

    def test_unmatched_previous_entries_are_dropped(self):
        self.assertEqual(merge_previous([{"id": None, "content": "x"}], [None], []), [])


class TestLoopIntegration(unittest.TestCase):
    """Runs src/rsi/loop.main() with training and sampling replaced by fakes."""

    TRAIN = [("a", "CCO", "qa"), ("b", "CCN", "qb"), ("c", "CCC", "qc"), ("d", "CCCl", "qd")]
    # instances each harvest re-solves, keyed by the iteration that produces the trace set
    SOLVED = {"iter_1": ["b", "c"], "iter_2": ["a", "b", "d"]}

    def setUp(self):
        import src.rsi.loop as loop
        self.loop, self.tmp = loop, tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.train = d / "train.txt"
        self.train.write_text("CID\tSMILES\tdescription\n" +
                              "".join(f"{i}\t{s}\t{q}\n" for i, s, q in self.TRAIN))
        self.seed = d / "seed.json"
        dump_json([{"conversations": [{"from": "human", "value": "qa"},
                                      {"from": "gpt", "value": "<think>s</think><answer>CCO</answer>"}]}],
                  str(self.seed))
        self.work = d / "work"
        self.calls = []
        self._orig = (loop.run, loop.run_harvest)
        loop.run, loop.run_harvest = self.fake_run, self.fake_harvest

    def tearDown(self):
        self.loop.run, self.loop.run_harvest = self._orig
        self.tmp.cleanup()

    def fake_run(self, cmd, desc):
        cmd = [str(c) for c in cmd]
        self.calls.append((Path(cmd[2]).name, cmd))
        final = Path(cmd[cmd.index("--output_dir") + 1]) / "final"
        final.mkdir(parents=True, exist_ok=True)
        (final / "model.safetensors").write_text("")

    def fake_harvest(self, args, model_dir, out_dir):
        self.calls.append(("harvest", [str(model_dir)]))
        solved = self.SOLVED[Path(out_dir).parent.name]
        rows = {i: (s, q) for i, s, q in self.TRAIN}
        entries = [{"id": i, "question": rows[i][1], "gt": rows[i][0], "content": f"<think>{i}</think>",
                    "pred": rows[i][0], "score": 1, "failure_times": 0} for i in solved]
        os.makedirs(out_dir, exist_ok=True)
        entries_path, meta_path = shard_paths(str(out_dir), 0)
        dump_json(entries, entries_path)
        dump_json({"items": 4, "accepted": len(solved), "unresolved": 4 - len(solved),
                   "samples_generated": 32, "max_attempts": 8, "samples_per_round": 8,
                   "require_format": True}, meta_path)
        return 1

    def run_loop(self):
        argv = sys.argv
        sys.argv = ["loop.py", "--base_model", "base", "--seed_data", str(self.seed),
                    "--train_data", str(self.train), "--work_dir", str(self.work), "--max_iterations", "5"]
        try:
            self.loop.main()
        finally:
            sys.argv = argv
        return load_json(str(self.work / "loop_log.json"))

    def test_merge_churn_stop_and_resume(self):
        log = self.run_loop()
        self.assertEqual(len(log), 2)
        t0, t1 = log
        # T=0: harvest solves b, c but not the seed instance a, which is carried over
        self.assertEqual(t0["harvest_churn"], {"new": 2, "lost": 1, "retained": 0, "net_growth": 1.0})
        self.assertEqual((t0["carried_over"], t0["next_trace_set"]["unique_ids"], t0["growth"]), (1, 3, 2.0))
        self.assertEqual(t0["samples_per_new_trace"], 16.0)
        # T=1: harvest re-solves a, b and adds d, misses c (carried over) -> all 4 covered
        self.assertEqual(t1["harvest_churn"]["new"], 1)
        self.assertEqual(t1["harvest_churn"]["lost"], 1)
        self.assertEqual(t1["next_trace_set"]["unique_ids"], 4)
        self.assertEqual(t1["stop"], "training set fully covered")
        # every SFT stage restarts from the base model; RL uses the raw training split
        sft1 = [c for name, c in self.calls if name == "sft_train.py"][1]
        self.assertEqual(sft1[sft1.index("--model_name") + 1], "base")
        rl0 = [c for name, c in self.calls if name == "grpo_train.py"][0]
        self.assertEqual(rl0[rl0.index("--data_path") + 1], str(self.train))
        self.assertEqual(rl0[rl0.index("--validity_weight") + 1], "0.0")
        merged = load_json(str(self.work / "iter_2" / "R.json"))
        self.assertEqual(sorted(e["id"] for e in merged), ["a", "b", "c", "d"])
        self.assertEqual([e["id"] for e in merged if e.get("from_previous")], ["c"])

        # re-running resumes: every stage is skipped and the log is unchanged
        self.calls.clear()
        self.assertEqual(self.run_loop(), log)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()

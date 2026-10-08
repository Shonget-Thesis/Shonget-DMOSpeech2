"""Small end-to-end check for pairing, blinding, and rating-key recovery."""

import csv
import random
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_results import analyze, make_packet, percentile, select_cases, write_csv


class AnalysisFlowTest(unittest.TestCase):
    def test_packet_and_rating_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "metrics.csv"
            rows = []
            for sample, speaker in (("1", "speaker_1"), ("2", "speaker_2")):
                for seed in ("42", "99"):
                    for method, wer in (("baseline", "0.2"), ("baseline_copy", "0.2"),
                                        ("fixed", "0.1"), ("proposed", "0.0")):
                        path = root / f"{sample}_{seed}_{method}.wav"
                        payload = b"same" if method in ("baseline", "baseline_copy") else method.encode()
                        path.write_bytes(payload)
                        rows.append({"sample_id": sample, "speaker_id": speaker, "seed": seed,
                                     "method": method, "wer": wer, "audio_file": path.name,
                                     "audio_seconds": "1", "inference_seconds": "0.5"})
            write_csv(results, ("sample_id", "speaker_id", "seed", "method", "wer",
                                "audio_file", "audio_seconds", "inference_seconds"), rows)
            packet = root / "packet"
            make_packet(Namespace(results=results, out_dir=packet, proposed="proposed",
                                  baseline="baseline", baseline_copy="baseline_copy",
                                  fixed="fixed", seed=2026, max_cases=4,
                                  naturalness_seed="42", naturalness_max_cases=None))
            key_file = packet / "analyst" / "answer_key.csv"
            with key_file.open(newline="", encoding="utf-8") as handle:
                key_rows = list(csv.DictReader(handle))
            self.assertEqual(len(key_rows), 8)
            self.assertFalse((packet / "listener" / "answer_key.csv").exists())
            ratings = root / "ratings.csv"
            write_csv(ratings, ("listener_id", "trial_id", "choice"), [
                {"listener_id": "rater_1", "trial_id": row["trial_id"],
                 "choice": "A" if row["method_a"] == "proposed" else "B"}
                for row in key_rows
            ])
            naturalness_key = packet / "analyst" / "naturalness_answer_key.csv"
            with naturalness_key.open(newline="", encoding="utf-8") as handle:
                naturalness_rows = list(csv.DictReader(handle))
            self.assertEqual(len(naturalness_rows), 2)
            self.assertEqual({row["seed"] for row in naturalness_rows}, {"42"})
            naturalness_ratings = root / "naturalness_ratings.csv"
            write_csv(naturalness_ratings, ("listener_id", "trial_id", "rating"), [
                {"listener_id": "rater_1", "trial_id": row["trial_id"],
                 "rating": "4" if row["speaker_id"] == "speaker_1" else "2"}
                for row in naturalness_rows
            ])
            report = root / "report.md"
            analyze(Namespace(results=results, ratings=ratings, key=key_file,
                              naturalness_ratings=naturalness_ratings,
                              naturalness_key=naturalness_key,
                              out=report, proposed="proposed", baseline="baseline",
                              fixed="fixed", baseline_copy="baseline_copy",
                              bootstrap=200, seed=2026, manifest=None, seeds=None))
            content = report.read_text(encoding="utf-8")
            self.assertIn("Duplicate baseline audit: **PASS**", content)
            self.assertIn("| proposed vs baseline | 4 | 2 | 4 | 1.0000 | [1.0000, 1.0000] |", content)
            self.assertIn("| proposed vs fixed | 4 | 2 | 4 | 1.0000 | [1.0000, 1.0000] |", content)
            self.assertIn("Baseline naturalness: **3.0000 / 5**", content)
            (root / "1_42_baseline_copy.wav").write_bytes(b"different")
            bad_packet = root / "bad_packet"
            with self.assertRaisesRegex(ValueError, "duplicate baseline audio differs"):
                make_packet(Namespace(results=results, out_dir=bad_packet, proposed="proposed",
                                      baseline="baseline", baseline_copy="baseline_copy",
                                      fixed="fixed", seed=2026, max_cases=4,
                                      naturalness_seed="42", naturalness_max_cases=2))
            self.assertFalse(bad_packet.exists())

    def test_percentile_at_exact_index(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.5), 2.0)

    def test_subset_balances_speaker_and_seed(self):
        cases = {
            (f"{speaker}-{seed}-{item}", speaker, seed): {}
            for speaker in ("spk1", "spk2")
            for seed in ("41", "42")
            for item in range(3)
        }
        chosen = select_cases(cases, 4, random.Random(2026))
        self.assertEqual({(speaker, seed) for _, speaker, seed in chosen},
                         {("spk1", "41"), ("spk1", "42"), ("spk2", "41"), ("spk2", "42")})
        self.assertEqual(chosen, select_cases(cases, 4, random.Random(2026)))
        with self.assertRaisesRegex(ValueError, "only 12 are available"):
            select_cases(cases, 13, random.Random(2026))


if __name__ == "__main__":
    unittest.main()

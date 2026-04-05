from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multi_agent_autoresearch.engine import AutoResearchEngine
from multi_agent_autoresearch.gsm8k_loop import GSM8KLoopConfig, GSM8KLoopRunner
from multi_agent_autoresearch.models import RunConfig


class SmokeTest(unittest.TestCase):
    def test_offline_run_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "artifacts"
            config = RunConfig(
                query="What makes a strong multi-agent autoresearch system?",
                output_dir=output_dir,
                search_provider="mock",
                max_rounds=2,
                max_subquestions=4,
                max_sources_per_question=3,
            )
            artifacts = AutoResearchEngine(config).run()
            self.assertGreaterEqual(len(artifacts.waves), 1)
            self.assertTrue((output_dir / "report.md").exists())
            self.assertTrue((output_dir / "report.json").exists())
            payload = json.loads((output_dir / "report.json").read_text())
            self.assertEqual(payload["config"]["search_provider"], "mock")
            self.assertGreaterEqual(len(payload["claims"]), 1)

    def test_localfs_run_reads_local_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            notes = root / "notes.txt"
            notes.write_text(
                "gsm8k grpo retained line is 0.505 and teacher anchor was tested before.\n",
                encoding="utf-8",
            )
            output_dir = root / "artifacts"
            config = RunConfig(
                query="What does the local gsm8k grpo history say about retained performance?",
                output_dir=output_dir,
                search_provider="localfs",
                local_roots=[str(notes)],
                max_rounds=1,
                max_subquestions=3,
                max_sources_per_question=3,
            )
            artifacts = AutoResearchEngine(config).run()
            self.assertGreaterEqual(len(artifacts.waves), 1)
            payload = json.loads((output_dir / "report.json").read_text())
            self.assertEqual(payload["config"]["search_provider"], "localfs")
            self.assertEqual(payload["config"]["local_roots"], [str(notes)])

    def test_localfs_ignores_virtualenv_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            noise_dir = root / ".venv" / "lib"
            noise_dir.mkdir(parents=True)
            (noise_dir / "noise.txt").write_text(
                "irrelevant metric noise from dependencies\n",
                encoding="utf-8",
            )
            notes = root / "research-results.tsv"
            notes.write_text(
                "iteration\tcommit\tmetric\tdelta\tguard\tstatus\tdescription\n"
                "0\tbase\t0.20\t0\t-\tbaseline\tbaseline run\n"
                "1\tkeep\t0.25\t0.05\tpass\tkeep\ttemperature 1.0 improved eval\n",
                encoding="utf-8",
            )
            output_dir = root / "artifacts"
            config = RunConfig(
                query="What does the local gsm8k grpo history say about retained performance?",
                output_dir=output_dir,
                search_provider="localfs",
                local_roots=[str(root)],
                max_rounds=1,
                max_subquestions=2,
                max_sources_per_question=3,
            )
            artifacts = AutoResearchEngine(config).run()
            evidence_urls = [item.source.url for wave in artifacts.waves for item in wave.evidence]
            self.assertFalse(any(".venv" in url for url in evidence_urls))

    def test_localfs_structures_autoresearch_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state = root / "autoresearch-state.json"
            state.write_text(
                json.dumps(
                    {
                        "state": {
                            "iteration": 15,
                            "best_metric": 0.25,
                            "current_metric": 0.25,
                            "best_iteration": 3,
                            "last_status": "pivot",
                        },
                        "supervisor": {
                            "recommended_action": "needs_human",
                            "last_reason": "Three strategic pivots were recorded without a keep.",
                        },
                        "updated_at": "2026-04-01T14:31:19Z",
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "artifacts"
            config = RunConfig(
                query="Summarize the gsm8k grpo progress",
                output_dir=output_dir,
                search_provider="localfs",
                local_roots=[str(state)],
                max_rounds=1,
                max_subquestions=2,
                max_sources_per_question=2,
            )
            artifacts = AutoResearchEngine(config).run()
            extracted = [item.extracted_fact for wave in artifacts.waves for item in wave.evidence]
            joined = "\n".join(extracted)
            self.assertIn("best_metric 0.25", joined)
            self.assertIn("recommended_action needs_human", joined)

    def test_localfs_reads_tsv_experiment_ledgers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = root / "research-results.tsv"
            ledger.write_text(
                "iteration\tcommit\tmetric\tdelta\tguard\tstatus\tdescription\n"
                "0\tbase\t0.234375\t0\t-\tbaseline\tbaseline run\n"
                "1\tkeep1\t0.25\t0.015625\tpass\tkeep\traise rollout sampling temperature from 0.8 to 1.0 improved fresh GSM8K eval\n"
                "2\t-\t0.25\t0\t-\tpivot\tnext work should prioritize optimizer or scheduler wiring\n",
                encoding="utf-8",
            )
            output_dir = root / "artifacts"
            config = RunConfig(
                query="What does the gsm8k grpo ledger say about retained progress and next steps?",
                output_dir=output_dir,
                search_provider="localfs",
                local_roots=[str(ledger)],
                max_rounds=1,
                max_subquestions=2,
                max_sources_per_question=2,
            )
            artifacts = AutoResearchEngine(config).run()
            extracted = [item.extracted_fact for wave in artifacts.waves for item in wave.evidence]
            joined = "\n".join(extracted)
            self.assertIn("best_metric 0.25", joined)
            self.assertIn("temperature from 0.8 to 1.0", joined)
            self.assertIn("optimizer or scheduler wiring", joined)

    def test_gsm8k_loop_runner_writes_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_env = root / "baseline.env"
            baseline_env.write_text(
                "\n".join(
                    [
                        "export DATASET_SOURCE='gsm8k'",
                        "export NUM_EVAL_SAMPLES='200'",
                        "export EVAL_NUM_CANDIDATES='12'",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_script = root / "fake_eval.py"
            fake_script.write_text(
                "from __future__ import annotations\n"
                "import json, os\n"
                "from pathlib import Path\n"
                "out = Path(os.environ['OUTPUT_DIR'])\n"
                "out.mkdir(parents=True, exist_ok=True)\n"
                "num = int(os.environ.get('EVAL_NUM_CANDIDATES', '12'))\n"
                "metric = 0.495 if num == 12 else 0.50\n"
                "count = int(metric * 200)\n"
                "payload = {'eval_before': {'exact_match_rate': metric, 'exact_match_count': count}, 'eval_after': {'exact_match_rate': metric, 'exact_match_count': count}}\n"
                "(out / 'run_summary.json').write_text(json.dumps(payload), encoding='utf-8')\n",
                encoding="utf-8",
            )
            output_dir = root / "loop_out"
            config = GSM8KLoopConfig(
                query="Improve gsm8k baseline",
                output_dir=output_dir,
                baseline_env_path=baseline_env,
                script_path=fake_script,
                local_roots=[str(root)],
                max_rounds=2,
                sync_script=str(root / "missing_sync.sh"),
                sync_repo=str(root / "missing_repo"),
            )
            state = GSM8KLoopRunner(config).run()
            self.assertTrue((output_dir / "loop_state.json").exists())
            payload = json.loads((output_dir / "loop_state.json").read_text())
            self.assertEqual(payload["best_exact_match_count"], 100)
            self.assertEqual(len(payload["iterations"]), 2)
            self.assertGreaterEqual(state.best_metric, 0.495)


if __name__ == "__main__":
    unittest.main()

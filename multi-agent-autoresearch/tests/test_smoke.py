from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multi_agent_autoresearch.engine import AutoResearchEngine
from multi_agent_autoresearch.gsm8k_loop import (
    GSM8KLoopConfig,
    GSM8KLoopRunner,
    GSM8KLoopState,
    _build_dynamic_proposals,
    _diagnose_run_summary,
    _infer_required_eval_dataset_name,
)
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
            self.assertEqual(payload["best_exact_match_count"], 99)
            self.assertEqual(len(payload["iterations"]), 3)
            self.assertGreaterEqual(state.best_metric, 0.495)
            self.assertTrue((output_dir / "director_summary.md").exists())

    def test_gsm8k_loop_runner_uses_baseline_runner_and_handles_early_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_env = root / "baseline.env"
            baseline_env.write_text("export DATASET_SOURCE='gsm8k'\n", encoding="utf-8")
            fake_script = root / "unused.py"
            fake_script.write_text("print('unused')\n", encoding="utf-8")
            fake_runner = root / "runner.sh"
            fake_runner.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "name=\"$1\"\n"
                "out=\"${OUTPUT_DIR:?}\"\n"
                "mkdir -p \"$out\"\n"
                "if [[ \"$name\" == *selector* ]]; then\n"
                "  printf 'observed_rate=0.375\\n' > \"$out/early_stop.txt\"\n"
                "  exit 10\n"
                "fi\n"
                "python3 - <<'PY'\n"
                "import json, os\n"
                "from pathlib import Path\n"
                "out = Path(os.environ['OUTPUT_DIR'])\n"
                "payload = {'eval_before': {'exact_match_rate': 0.515, 'exact_match_count': 103, 'rows': []}, 'eval_after': {'exact_match_rate': 0.515, 'exact_match_count': 103, 'rows': []}}\n"
                "(out / 'run_summary.json').write_text(json.dumps(payload), encoding='utf-8')\n"
                "PY\n",
                encoding="utf-8",
            )
            fake_runner.chmod(0o755)
            output_dir = root / "loop_out"
            config = GSM8KLoopConfig(
                query="Improve gsm8k baseline",
                output_dir=output_dir,
                baseline_env_path=baseline_env,
                script_path=fake_script,
                runner_path=fake_runner,
                local_roots=[str(root)],
                max_rounds=1,
                enable_research_wave=False,
                sync_script=str(root / "missing_sync.sh"),
                sync_repo=str(root / "missing_repo"),
            )
            state = GSM8KLoopRunner(config).run()
            self.assertEqual(len(state.iterations), 2)
            self.assertEqual(state.iterations[1].status, "early_stop")
            self.assertIn("runner_early_stop", state.iterations[1].notes)

    def test_gsm8k_loop_runner_resumes_existing_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_env = root / "baseline.env"
            baseline_env.write_text("export DATASET_SOURCE='gsm8k'\n", encoding="utf-8")
            fake_script = root / "fake_eval.py"
            fake_script.write_text(
                "from __future__ import annotations\n"
                "import json, os\n"
                "from pathlib import Path\n"
                "out = Path(os.environ['OUTPUT_DIR'])\n"
                "out.mkdir(parents=True, exist_ok=True)\n"
                "temp = os.environ.get('EVAL_RERANK_TEMPERATURE', '0.65')\n"
                "metric = 0.52 if temp == '0.7' else 0.515\n"
                "count = int(metric * 200)\n"
                "payload = {\n"
                "  'eval_before': {'exact_match_rate': metric, 'exact_match_count': count, 'rows': []},\n"
                "  'eval_after': {'exact_match_rate': metric, 'exact_match_count': count, 'rows': []},\n"
                "}\n"
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
                max_rounds=1,
                enable_research_wave=False,
                sync_script=str(root / "missing_sync.sh"),
                sync_repo=str(root / "missing_repo"),
            )
            GSM8KLoopRunner(config).run()
            resumed = GSM8KLoopRunner(
                GSM8KLoopConfig(
                    query="Improve gsm8k baseline",
                    output_dir=output_dir,
                    baseline_env_path=baseline_env,
                    script_path=fake_script,
                    local_roots=[str(root)],
                    max_rounds=2,
                    enable_research_wave=False,
                    sync_script=str(root / "missing_sync.sh"),
                    sync_repo=str(root / "missing_repo"),
                )
            ).run()
            self.assertGreaterEqual(len(resumed.iterations), 3)
            self.assertGreaterEqual(resumed.best_metric, 0.515)

    def test_infer_required_eval_dataset_name_prefers_train_validation_split(self) -> None:
        env = {
            "DATASET_SOURCE": "gsm8k",
            "DATASET_SPLIT": "train",
            "TRAIN_VALIDATION_MOD": "5",
            "TRAIN_VALIDATION_BUCKET": "2",
        }
        self.assertEqual(
            _infer_required_eval_dataset_name(env),
            "gsm8k_train_validation_mod5_bucket2",
        )

    def test_diagnosis_reads_error_attribution_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = Path(tmpdir) / "run_summary.json"
            payload = {
                "eval_after": {
                    "exact_match_rate": 0.5,
                    "exact_match_count": 100,
                    "rows": [],
                    "error_attribution": {
                        "wrong::rate_ratio_unit_chain::large_numeric_error": 9,
                        "wrong::percentage_discount_growth::large_numeric_error": 7,
                        "correct::other": 11,
                    },
                },
                "eval_dataset_name": "gsm8k_train_validation_mod5_bucket0",
            }
            summary.write_text(json.dumps(payload), encoding="utf-8")
            diagnosis = _diagnose_run_summary(summary, "eval_after")
            self.assertEqual(
                diagnosis["top_wrong_patterns"][0],
                "wrong::rate_ratio_unit_chain::large_numeric_error",
            )
            self.assertEqual(
                diagnosis["eval_dataset_name"],
                "gsm8k_train_validation_mod5_bucket0",
            )

    def test_dynamic_proposals_prioritize_validation_failure_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_env = root / "baseline.env"
            baseline_env.write_text(
                "\n".join(
                    [
                        "export DATASET_SOURCE='gsm8k'",
                        "export DATASET_SPLIT='train'",
                        "export TRAIN_VALIDATION_MOD='5'",
                        "export TRAIN_VALIDATION_BUCKET='0'",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = GSM8KLoopConfig(
                query="Improve gsm8k baseline",
                output_dir=root / "loop_out",
                baseline_env_path=baseline_env,
                script_path=root / "fake.py",
                local_roots=[str(root)],
                max_rounds=1,
                enable_research_wave=False,
                sync_script=str(root / "missing_sync.sh"),
                sync_repo=str(root / "missing_repo"),
            )
            state = GSM8KLoopState(
                config=config,
                started_at="2026-04-10T00:00:00Z",
                best_metric=0.5,
                best_exact_match_count=100,
                best_label="baseline-env",
                best_output_dir="",
                current_env={},
                external_history={},
            )
            state.latest_diagnosis = {
                "exact_match_rate": 0.5,
                "bottom_slices": ["percentage", "rate_or_ratio"],
                "top_wrong_patterns": [
                    "wrong::rate_ratio_unit_chain::large_numeric_error",
                    "wrong::percentage_discount_growth::large_numeric_error",
                ],
                "strict_xml_rate": 1.0,
                "numeric_answer_rate": 1.0,
                "gap": 0.08,
                "correctness_reward_mean": 3.0,
                "distance_reward_mean": 0.25,
            }
            proposals = _build_dynamic_proposals(state)
            top_labels = [item.label for item in proposals[:3]]
            self.assertIn("selector_numc12_verifier03_expand", top_labels)
            self.assertIn("data_quality_strict_065_min22_validation", top_labels)


if __name__ == "__main__":
    unittest.main()

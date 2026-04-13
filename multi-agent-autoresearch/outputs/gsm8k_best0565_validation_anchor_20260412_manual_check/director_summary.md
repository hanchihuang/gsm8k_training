# GSM8K Director Summary

- started_at: 2026-04-12T06:00:37+00:00
- best_metric: 0.613
- best_exact_match_count: 111
- best_label: baseline
- best_output_dir: /home/user/图片/gsm8k_training_repo/gsm8k_improved/confirm200_neartop_override_explicit_20260406_validation_run_20260410

## External History

- root: /home/user/图片/gsm8k_improved
- completed_runs: 3
- early_stops: 2
- best_metric: 0.000
- best_dir: -

## Current Diagnosis

- metric_section: eval_after
- eval_dataset_name: gsm8k_train_validation_mod5_bucket0
- exact_match_rate: 0.608
- pass1: 0.608
- pass8: 0.608
- selector_gap: 0.000
- strict_xml_rate: 0.994
- numeric_answer_rate: 0.978
- correctness_reward_mean: 3.646
- distance_reward_mean: 0.032
- bottom_slices: percentage, rate_or_ratio, multi_number
- top_wrong_patterns: wrong::rate_ratio_unit_chain::large_numeric_error, wrong::percentage_discount_growth::large_numeric_error, wrong::rate_ratio_unit_chain::medium_numeric_error, wrong::multiplicative_relation::large_numeric_error
- regression_buckets: long_tail_reasoning, noisy_numeric_failures, answer_extraction_noise

## Recent Iterations

- iter -1: baseline [baseline/baseline] status=keep metric=0.613 notes=baseline_summary=/home/user/图片/gsm8k_training_repo/gsm8k_improved/confirm200_neartop_override_explicit_20260406_validation_run_20260410/run_summary.json; seed-baseline
- iter 0: data_quality_strict_065_min22_validation [data_quality_train/write_policy] status=discard metric=0.304 notes=-
- iter 1: selector_verifier035_margin026 [selector_retrieval/retrieval] status=discard metric=0.608 notes=-

## Next Hypotheses

- selector_verifier035_margin026 [selector_retrieval/retrieval] priority=2.60: Conservative retrieval ablation: increase verifier influence without reopening candidate expansion.
- selector_consensus_up_paircount055 [selector_retrieval/retrieval] priority=2.25: Retrieval ablation: favor agreement structure on the anchored non-expand selector path.
- selector_reranker_margin_tight025 [selector_retrieval/retrieval] priority=2.25: Retrieval ablation: tighten near-top override and verifier tie behavior to reduce wrong near-top jumps.
- prompt_compact_xml_192 [compression_prompt/compression] priority=-0.60: Compression ablation: shorten prompt budget while preserving the XML answer path.
- prompt_compact_xml_160 [compression_prompt/compression] priority=-0.60: Compression ablation: test whether a more aggressively compact context reduces long-tail reasoning noise.

## Experiment Log

- iter 1: selector_verifier035_margin026 axis=retrieval status=discard delta=-0.006 buckets=long_tail_reasoning, noisy_numeric_failures, answer_extraction_noise

# GSM8K Director Summary

- started_at: 2026-04-12T13:58:40+00:00
- anchor_test_metric: 0.565
- anchor_config: /home/user/图片/gsm8k_training_repo/gsm8k_improved/best_seed_confirm200_neartop_override_explicit_20260406.env
- best_metric: 0.663
- best_exact_match_count: 120
- best_label: reward_reduced_v2_train
- best_output_dir: /home/user/图片/gsm8k_training_repo/multi-agent-autoresearch/outputs/gsm8k_best0565_official_validation_anchor_20260412_215801/runs/iter_005_reward_reduced_v2_train

## External History

- root: /home/user/图片/gsm8k_improved
- completed_runs: 157
- early_stops: 2
- best_metric: 0.567
- best_dir: /home/user/图片/gsm8k_improved/autoresearch_evalonly_verifier_c05_from0485_scout30_20260331

## Current Diagnosis

- metric_section: eval_after
- eval_dataset_name: gsm8k_train_validation_mod5_bucket0
- exact_match_rate: 0.530
- pass1: 0.530
- pass8: 0.530
- selector_gap: 0.000
- strict_xml_rate: 1.000
- numeric_answer_rate: 0.989
- correctness_reward_mean: 3.182
- distance_reward_mean: 0.036
- bottom_slices: percentage, rate_or_ratio, basic_arithmetic
- top_wrong_patterns: wrong::rate_ratio_unit_chain::large_numeric_error, wrong::percentage_discount_growth::large_numeric_error, wrong::rate_ratio_unit_chain::medium_numeric_error, wrong::multiplicative_relation::large_numeric_error
- regression_buckets: long_tail_reasoning, noisy_numeric_failures, answer_extraction_noise

## Recent Iterations

- iter 2: prompt_compact_xml_160 [compression_prompt/compression] status=early_stop metric=0.000 notes=runner_early_stop; early_stop_sample=40; exact_matches=18; observed_rate=0.45; threshold=0.5149999999999999
- iter 3: selector_consensus_up_paircount055 [selector_retrieval/retrieval] status=discard metric=0.608 notes=-
- iter 4: selector_reranker_margin_tight025 [selector_retrieval/retrieval] status=discard metric=0.602 notes=-
- iter 5: reward_reduced_v2_train [reward_train/write_policy] status=keep metric=0.663 notes=improved-best
- iter 6: data_quality_strict_065_min22_validation [data_quality_train/write_policy] status=discard metric=0.530 notes=-

## Next Hypotheses

- data_quality_balanced_075_train [data_quality_train/write_policy] priority=-97.00: Use moderate data-quality filtering near the successful P15 regime instead of strict pruning.
- data_quality_balanced_07_min16_train [data_quality_train/write_policy] priority=-97.00: Stay close to the 0.56 line but add a mild cutoff to remove only the lowest-value training questions.

## Experiment Log

- iter 2: prompt_compact_xml_160 axis=compression status=early_stop delta=-0.613 buckets=-
- iter 3: selector_consensus_up_paircount055 axis=retrieval status=discard delta=-0.006 buckets=long_tail_reasoning, noisy_numeric_failures, answer_extraction_noise
- iter 4: selector_reranker_margin_tight025 axis=retrieval status=discard delta=-0.011 buckets=long_tail_reasoning, noisy_numeric_failures, answer_extraction_noise
- iter 5: reward_reduced_v2_train axis=write_policy status=keep delta=0.050 buckets=long_tail_reasoning, noisy_numeric_failures, format_instability, answer_extraction_noise
- iter 6: data_quality_strict_065_min22_validation axis=write_policy status=discard delta=-0.133 buckets=long_tail_reasoning, noisy_numeric_failures, answer_extraction_noise

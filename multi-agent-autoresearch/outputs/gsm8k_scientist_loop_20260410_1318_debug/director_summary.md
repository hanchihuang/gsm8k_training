# GSM8K Director Summary

- started_at: 2026-04-10T05:25:16+00:00
- best_metric: 0.515
- best_exact_match_count: 103
- best_label: baseline
- best_output_dir: /home/user/图片/gsm8k_training_repo/multi-agent-autoresearch/outputs/gsm8k_scientist_loop_20260410_1318_debug/runs/iter_-01_baseline

## External History

- root: /home/user/图片/gsm8k_improved
- completed_runs: 164
- early_stops: 2
- best_metric: 0.800
- best_dir: /home/user/图片/gsm8k_improved/qwen25_3b_evalonly_scout60_20260331

## Current Diagnosis

- metric_section: eval_after
- exact_match_rate: 0.460
- pass1: 0.460
- pass8: 0.000
- selector_gap: -0.460
- strict_xml_rate: 1.000
- numeric_answer_rate: 0.975
- correctness_reward_mean: 2.760
- distance_reward_mean: 0.046
- bottom_slices: percentage, rate_or_ratio, multi_number

## Recent Iterations

- iter -1: baseline [baseline] status=keep metric=0.515 notes=seed-baseline
- iter 0: prompt_answer_first [prompt] status=early_stop metric=0.000 notes=runner_early_stop; early_stop_sample=40; exact_matches=19; observed_rate=0.475; threshold=0.5149999999999999
- iter 1: selector_numc12_top_p095 [selector] status=early_stop metric=0.000 notes=runner_early_stop; early_stop_sample=40; exact_matches=20; observed_rate=0.5; threshold=0.5149999999999999
- iter 2: reward_reduced_v1_train [reward_train] status=discard metric=0.460 notes=-

## Next Hypotheses

- reward_reduced_v2_train [reward_train] priority=0.00: Push reward harder toward exact correctness after reward-down v1 if the issue persists.
- selector_numc12_temp07_top_p095 [selector] priority=-5.20: Broaden candidate pool diversity while retaining the 0.565 rerank stack.
- selector_numc12_verifier025 [selector] priority=-5.20: Lean harder on verifier score when selector instability dominates.
- selector_expand_profiles [selector] priority=-5.20: Use profile expansion when the pool has answers but lacks diversity on weak slices.

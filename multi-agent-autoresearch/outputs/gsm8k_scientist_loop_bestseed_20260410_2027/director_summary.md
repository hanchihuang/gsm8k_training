# GSM8K Director Summary

- started_at: 2026-04-10T12:23:33+00:00
- best_metric: 0.567
- best_exact_match_count: 113
- best_label: external_best
- best_output_dir: /home/user/图片/gsm8k_improved/autoresearch_evalonly_verifier_c05_from0485_scout30_20260331

## External History

- root: /home/user/图片/gsm8k_improved
- completed_runs: 157
- early_stops: 2
- best_metric: 0.567
- best_dir: /home/user/图片/gsm8k_improved/autoresearch_evalonly_verifier_c05_from0485_scout30_20260331

## Current Diagnosis

- metric_section: eval_after
- exact_match_rate: 0.515
- pass1: 0.515
- pass8: 0.515
- selector_gap: 0.000
- strict_xml_rate: 0.995
- numeric_answer_rate: 0.975
- correctness_reward_mean: 3.090
- distance_reward_mean: 0.025
- bottom_slices: percentage, rate_or_ratio, difference

## Recent Iterations

- iter -1: baseline [baseline] status=keep metric=0.515 notes=seed-baseline

## Next Hypotheses

- data_quality_balanced_075_train [data_quality_train] priority=1.70: Use moderate data-quality filtering near the successful P15 regime instead of strict pruning.
- data_quality_balanced_07_min16_train [data_quality_train] priority=1.70: Stay close to the 0.56 line but add a mild cutoff to remove only the lowest-value training questions.
- selector_margin_loosened [selector] priority=1.55: Loosen answer aggregation margin slightly while increasing pair-count support.
- selector_numc12_top_p095 [selector] priority=1.55: Increase candidate coverage when pass@8 is above pass@1.
- selector_numc12_temp07_top_p095 [selector] priority=1.55: Broaden candidate pool diversity while retaining the 0.565 rerank stack.

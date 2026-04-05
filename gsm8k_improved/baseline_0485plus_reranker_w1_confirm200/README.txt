Baseline 0.485+ Reranker W1 Confirm200
======================================

This directory freezes the confirm200 evaluation line that reaches at least
97/200 (= 0.485) on eval_before with:

  /home/user/图片/outputs_llama3_1_grpo_gsm8k_rerank_eval/adapter
  + verifier bundle
  + reranker v2 pointwise bundle
  + RERANKER_SCORE_WEIGHT=1.0

Reference fresh rerun:

  /home/user/图片/gsm8k_improved/recheck_reranker_v2_w1_confirm200_20260405

Target metric:

  eval_before.exact_match_count >= 97 / 200

Files:

- baseline.env
  Frozen environment settings.

- run_baseline.sh
  Runs into a timestamped output directory under:
  /home/user/图片/gsm8k_improved/baseline_0485plus_reranker_w1_confirm200/runs

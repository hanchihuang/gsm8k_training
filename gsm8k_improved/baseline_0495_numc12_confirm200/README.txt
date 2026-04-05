Baseline 0.495 NumC12 Confirm200
================================

This directory freezes the confirm200 evaluation line that reaches:

  eval_before.exact_match_count = 99 / 200
  eval_before.exact_match_rate = 0.495

Configuration:

  /home/user/图片/outputs_llama3_1_grpo_gsm8k_rerank_eval/adapter
  + verifier bundle
  + reranker v2 pointwise bundle
  + RERANKER_SCORE_WEIGHT=1.0
  + EVAL_NUM_CANDIDATES=12

Reference run:

  /home/user/图片/gsm8k_improved/confirm200_w1_numc12_20260405

Files:

- baseline.env
  Frozen environment settings.

- run_baseline.sh
  Runs into a timestamped output directory under:
  /home/user/图片/gsm8k_improved/baseline_0495_numc12_confirm200/runs

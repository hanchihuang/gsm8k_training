Baseline 0.485 Confirm200
=========================

This directory freezes the reproducible 0.485 eval_before baseline that was
observed for:

  /home/user/图片/outputs_llama3_1_grpo_gsm8k_rerank_eval/adapter

Authoritative reference run:

  /home/user/图片/gsm8k_improved/evalonly_reranker_v2_pointwise_c2_w1_confirm200_20260404/run_summary.json

Reference metric:

  eval_before.exact_match_rate = 0.485
  eval_before.exact_match_count = 97 / 200

Important:

1. This baseline is the current mainline worth iterating on.
2. Historical 0.505 supportscore_keep is a different adapter/artifact line and
   is not currently reproducible under today's stack.
3. New experiments should start from this fixed command unless explicitly
   testing a different adapter family.

Files:

- baseline.env
  Frozen environment settings for the 0.485 line.

- run_baseline.sh
  Runs the baseline into a timestamped output directory under:
  /home/user/图片/gsm8k_improved/baseline_0485_confirm200/runs

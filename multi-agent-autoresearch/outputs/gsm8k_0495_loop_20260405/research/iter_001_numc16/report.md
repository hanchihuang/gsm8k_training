# Research Report

## Query

How should the GSM8K confirm200 line improve beyond the current retained baseline?
Current best metric: 0.000 (0/200).
Next candidate proposal label: numc16.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history.

## Plan

- What sub-problems must be solved to answer: How should the GSM8K confirm200 line improve beyond the current retained baseline?
Current best metric: 0.000 (0/200).
Next candidate proposal label: numc16.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history.?
- What evidence would make the answer to 'How should the GSM8K confirm200 line improve beyond the current retained baseline?
Current best metric: 0.000 (0/200).
Next candidate proposal label: numc16.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history.' trustworthy?
- What are the strongest design patterns related to: How should the GSM8K confirm200 line improve beyond the current retained baseline?
Current best metric: 0.000 (0/200).
Next candidate proposal label: numc16.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history.?
- What failure modes or blind spots appear in systems for: How should the GSM8K confirm200 line improve beyond the current retained baseline?
Current best metric: 0.000 (0/200).
Next candidate proposal label: numc16.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history.?

## Verified Claims

- "prompt_text": "Mode: loop\nContinue improving GSM8K exact match using codex-autoresearch from the current retained best of 0.505 on test[:200]  
  status: `supported` | support: `1.0` | evidence: /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json
- Prioritize rerank and inference-time changes in llama3_1_(8b)_grpo.py first; training-side changes are allowed later only if rerank lines stall  
  status: `supported` | support: `1.0` | evidence: /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json
- Use mechanical verification only, keep formatting metrics stable, and log every completed experiment before th  
  status: `supported` | support: `1.0` | evidence: /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json
- research results; baseline_metric 0.47; best_metric 0.5; best_status keep; best_description [labels: verifier_guided_selection, slice_aware_shaping, failure_slice_eval_gate, retained_control] [verifier_guided_selection] reused the retained eval-only mainline but switched answer-group and expansion thresholds to be failure-slice aware from the confirm200 summary: percentage and rate_or_ratio questions keep the aggressive verifier/expansion path, while stronger slices require a more conservative aggregate override  
  status: `supported` | support: `1.0` | evidence: /home/user/图片/research-results.tsv, /home/user/图片/research-results.tsv, /home/user/图片/research-results.tsv
- The manifest scout verify improved from 0.47 to 0.50 (15/30) with answer_tag/strict_xml at 0.97/0.97, so this slice-conditioned eval gate becomes the new retained control branch.; keep_count 1; latest_pivot [labels: pivot, verifier_guided_selection, slice_aware_shaping, failure_conditioned_gate] [PIVOT] abandoning the rollback hypothesis that the retained-mainline regression came from the implicit pointwise reranker bonus  
  status: `supported` | support: `1.0` | evidence: /home/user/图片/research-results.tsv, /home/user/图片/research-results.tsv, /home/user/图片/research-results.tsv
- Explicitly zeroing that bonus still regressed scout30 to 0.43, so the next strategy family is a failure-conditioned verifier/slice gate that uses offline candidate-history signals to decide when expansion or verifier overrides are allowed, instead of removing whole score components from the control path.; latest_status refine  
  status: `supported` | support: `1.0` | evidence: /home/user/图片/research-results.tsv, /home/user/图片/research-results.tsv, /home/user/图片/research-results.tsv

## Critic Verdict

- decision: `continue`
- rationale: The evidence is still narrow or under-verified.
- follow-up: Find more diverse sources that answer: How should the GSM8K confirm200 line improve beyond the current retained baseline?
Current best metric: 0.000 (0/200).
Next candidate proposal label: numc16.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history. from different communities or toolchains.
- follow-up: What verification and evaluation mechanisms are used in systems addressing: How should the GSM8K confirm200 line improve beyond the current retained baseline?
Current best metric: 0.000 (0/200).
Next candidate proposal label: numc16.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history.?
- follow-up: How do robust systems persist lessons, failed attempts, or shared knowledge for: How should the GSM8K confirm200 line improve beyond the current retained baseline?
Current best metric: 0.000 (0/200).
Next candidate proposal label: numc16.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history.?

## Lessons

- "prompt_text": "Mode: loop\nContinue improving GSM8K exact match using codex-autoresearch from the current retained best of 0.505 on test[:200]
- Prioritize rerank and inference-time changes in llama3_1_(8b)_grpo.py first; training-side changes are allowed later only if rerank lines stall
- Use mechanical verification only, keep formatting metrics stable, and log every completed experiment before th
- research results; baseline_metric 0.47; best_metric 0.5; best_status keep; best_description [labels: verifier_guided_selection, slice_aware_shaping, failure_slice_eval_gate, retained_control] [verifier_guided_selection] reused the retained eval-only mainline but switched answer-group and expansion thresholds to be failure-slice aware from the confirm200 summary: percentage and rate_or_ratio questions keep the aggressive verifier/expansion path, while stronger slices require a more conservative aggregate override
- The manifest scout verify improved from 0.47 to 0.50 (15/30) with answer_tag/strict_xml at 0.97/0.97, so this slice-conditioned eval gate becomes the new retained control branch.; keep_count 1; latest_pivot [labels: pivot, verifier_guided_selection, slice_aware_shaping, failure_conditioned_gate] [PIVOT] abandoning the rollback hypothesis that the retained-mainline regression came from the implicit pointwise reranker bonus
- Explicitly zeroing that bonus still regressed scout30 to 0.43, so the next strategy family is a failure-conditioned verifier/slice gate that uses offline candidate-history signals to decide when expansion or verifier overrides are allowed, instead of removing whole score components from the control path.; latest_status refine

## Sources

- [gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json](/home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json)
- [research-results.tsv](/home/user/图片/research-results.tsv)
- [gsm8k_improved/autoresearch_scout_iter90_dr_grpo_masktrunc/run_summary.json](/home/user/图片/gsm8k_improved/autoresearch_scout_iter90_dr_grpo_masktrunc/run_summary.json)
- [gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/research-results.tsv](/home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/research-results.tsv)
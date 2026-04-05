# Research Report

## Query

How should the GSM8K confirm200 line improve beyond the current retained baseline?
Current best metric: 0.000 (0/200).
Next candidate proposal label: numc12_top_p095.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history.

## Plan

- What sub-problems must be solved to answer: How should the GSM8K confirm200 line improve beyond the current retained baseline?
Current best metric: 0.000 (0/200).
Next candidate proposal label: numc12_top_p095.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history.?
- What evidence would make the answer to 'How should the GSM8K confirm200 line improve beyond the current retained baseline?
Current best metric: 0.000 (0/200).
Next candidate proposal label: numc12_top_p095.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history.' trustworthy?
- What are the strongest design patterns related to: How should the GSM8K confirm200 line improve beyond the current retained baseline?
Current best metric: 0.000 (0/200).
Next candidate proposal label: numc12_top_p095.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history.?
- What failure modes or blind spots appear in systems for: How should the GSM8K confirm200 line improve beyond the current retained baseline?
Current best metric: 0.000 (0/200).
Next candidate proposal label: numc12_top_p095.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history.?

## Verified Claims

- "prompt_text": "Mode: loop\nContinue improving GSM8K exact match using codex-autoresearch from the current retained best of 0.505 on test[:200]  
  status: `supported` | support: `1.0` | evidence: /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json
- Prioritize rerank and inference-time changes in llama3_1_(8b)_grpo.py first; training-side changes are allowed later only if rerank lines stall  
  status: `supported` | support: `1.0` | evidence: /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json
- Use mechanical verification only, keep formatting metrics stable, and log every completed experiment before th  
  status: `supported` | support: `1.0` | evidence: /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json
- "experiment_hypothesis": "Switching the retained mask-truncated continuation scout from dapo to dr_grpo should reduce length bias on clipped GSM8K rollouts and improve exact match over the retained-control neighborhood.",  
  status: `supported` | support: `1.0` | evidence: /home/user/图片/gsm8k_improved/autoresearch_scout_iter90_dr_grpo_masktrunc/run_summary.json, /home/user/图片/gsm8k_improved/autoresearch_scout_iter90_dr_grpo_masktrunc/run_summary.json, /home/user/图片/gsm8k_improved/autoresearch_scout_iter90_dr_grpo_masktrunc/run_summary.json
- research results; baseline_metric 0.485; best_metric 0.485; best_status baseline; best_description Authoritative baseline on 2026-03-27: rerank eval adapter, test[:200], EVAL_USE_CONFIDENCE_RERANK=1, EVAL_NUM_CANDIDATES=8 -> exact_match_rate 0.485 with answer_tag_rate 0.99 and strict_xml_rate 0.98; keep_count 0; latest_status discard  
  status: `supported` | support: `1.0` | evidence: /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/research-results.tsv, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/research-results.tsv, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/research-results.tsv
- "experiment_hypothesis": "The mask_truncated_completions continuation recipe that won the scout gate should retain enough signal under a matched 40-step run to improve 200-sample exact match over the kept 0.475 verifier-rerank neighborhood.",  
  status: `supported` | support: `1.0` | evidence: /home/user/图片/gsm8k_improved/autoresearch_confirm200_masktrunc_iter75c/run_summary.json, /home/user/图片/gsm8k_improved/autoresearch_confirm200_masktrunc_iter75c/run_summary.json, /home/user/图片/gsm8k_improved/autoresearch_confirm200_masktrunc_iter75c/run_summary.json

## Critic Verdict

- decision: `continue`
- rationale: The evidence is still narrow or under-verified.
- follow-up: Find more diverse sources that answer: How should the GSM8K confirm200 line improve beyond the current retained baseline?
Current best metric: 0.000 (0/200).
Next candidate proposal label: numc12_top_p095.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history. from different communities or toolchains.
- follow-up: What verification and evaluation mechanisms are used in systems addressing: How should the GSM8K confirm200 line improve beyond the current retained baseline?
Current best metric: 0.000 (0/200).
Next candidate proposal label: numc12_top_p095.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history.?
- follow-up: How do robust systems persist lessons, failed attempts, or shared knowledge for: How should the GSM8K confirm200 line improve beyond the current retained baseline?
Current best metric: 0.000 (0/200).
Next candidate proposal label: numc12_top_p095.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history.?

## Lessons

- "prompt_text": "Mode: loop\nContinue improving GSM8K exact match using codex-autoresearch from the current retained best of 0.505 on test[:200]
- Prioritize rerank and inference-time changes in llama3_1_(8b)_grpo.py first; training-side changes are allowed later only if rerank lines stall
- Use mechanical verification only, keep formatting metrics stable, and log every completed experiment before th
- "experiment_hypothesis": "Switching the retained mask-truncated continuation scout from dapo to dr_grpo should reduce length bias on clipped GSM8K rollouts and improve exact match over the retained-control neighborhood.",
- research results; baseline_metric 0.485; best_metric 0.485; best_status baseline; best_description Authoritative baseline on 2026-03-27: rerank eval adapter, test[:200], EVAL_USE_CONFIDENCE_RERANK=1, EVAL_NUM_CANDIDATES=8 -> exact_match_rate 0.485 with answer_tag_rate 0.99 and strict_xml_rate 0.98; keep_count 0; latest_status discard
- "experiment_hypothesis": "The mask_truncated_completions continuation recipe that won the scout gate should retain enough signal under a matched 40-step run to improve 200-sample exact match over the kept 0.475 verifier-rerank neighborhood.",

## Sources

- [gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json](/home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json)
- [gsm8k_improved/autoresearch_scout_iter90_dr_grpo_masktrunc/run_summary.json](/home/user/图片/gsm8k_improved/autoresearch_scout_iter90_dr_grpo_masktrunc/run_summary.json)
- [gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/research-results.tsv](/home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/research-results.tsv)
- [gsm8k_improved/autoresearch_scout_iter77_schema_archetypes/run_summary.json](/home/user/图片/gsm8k_improved/autoresearch_scout_iter77_schema_archetypes/run_summary.json)
- [gsm8k_improved/autoresearch_confirm200_masktrunc_iter75c/run_summary.json](/home/user/图片/gsm8k_improved/autoresearch_confirm200_masktrunc_iter75c/run_summary.json)
- [gsm8k_improved/verifier_temp_soft_probe20/run_summary.json](/home/user/图片/gsm8k_improved/verifier_temp_soft_probe20/run_summary.json)
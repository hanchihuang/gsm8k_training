# Research Report

## Query

Push the Qwen2.5-0.5B GSM8K confirm200 line beyond 0.565 by prioritizing candidate_distribution and longer-GRPO continuation routes. Do not spend more iterations on rerank_tuning unless a training family beats the baseline. Focus on prompt replay, teacher replay, and longer guarded continuation on percentage and rate_or_ratio.
Current best metric: 0.480 (96/200).
Next candidate proposal label: pair045_neartop_expand.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history.

## Plan

- What sub-problems must be solved to answer: Push the Qwen2.5-0.5B GSM8K confirm200 line beyond 0.565 by prioritizing candidate_distribution and longer-GRPO continuation routes. Do not spend more iterations on rerank_tuning unless a training family beats the baseline. Focus on prompt replay, teacher replay, and longer guarded continuation on percentage and rate_or_ratio.
Current best metric: 0.480 (96/200).
Next candidate proposal label: pair045_neartop_expand.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history.?
- What evidence would make the answer to 'Push the Qwen2.5-0.5B GSM8K confirm200 line beyond 0.565 by prioritizing candidate_distribution and longer-GRPO continuation routes. Do not spend more iterations on rerank_tuning unless a training family beats the baseline. Focus on prompt replay, teacher replay, and longer guarded continuation on percentage and rate_or_ratio.
Current best metric: 0.480 (96/200).
Next candidate proposal label: pair045_neartop_expand.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history.' trustworthy?
- What are the strongest design patterns related to: Push the Qwen2.5-0.5B GSM8K confirm200 line beyond 0.565 by prioritizing candidate_distribution and longer-GRPO continuation routes. Do not spend more iterations on rerank_tuning unless a training family beats the baseline. Focus on prompt replay, teacher replay, and longer guarded continuation on percentage and rate_or_ratio.
Current best metric: 0.480 (96/200).
Next candidate proposal label: pair045_neartop_expand.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history.?
- What failure modes or blind spots appear in systems for: Push the Qwen2.5-0.5B GSM8K confirm200 line beyond 0.565 by prioritizing candidate_distribution and longer-GRPO continuation routes. Do not spend more iterations on rerank_tuning unless a training family beats the baseline. Focus on prompt replay, teacher replay, and longer guarded continuation on percentage and rate_or_ratio.
Current best metric: 0.480 (96/200).
Next candidate proposal label: pair045_neartop_expand.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history.?

## Verified Claims

- "prompt_text": "Mode: loop\nContinue improving GSM8K exact match using codex-autoresearch from the current retained best of 0.505 on test[:200]  
  status: `supported` | support: `1.0` | evidence: /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json
- Prioritize rerank and inference-time changes in llama3_1_(8b)_grpo.py first; training-side changes are allowed later only if rerank lines stall  
  status: `supported` | support: `1.0` | evidence: /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json
- Use mechanical verification only, keep formatting metrics stable, and log every completed experiment before th  
  status: `supported` | support: `1.0` | evidence: /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json
- "experiment_risk": "if policy drift is caused by GRPO continuation itself rather than synthetic volume, cutting augmentation from 16 to 8 will not recover the retained verifier-rerank neighborhood",  
  status: `supported` | support: `1.0` | evidence: /home/user/图片/gsm8k_improved/autoresearch_scout_cached_teacher_lowaug_iter50/run_summary.json, /home/user/图片/gsm8k_training_repo/gsm8k_improved/autoresearch_scout_cached_teacher_lowaug_iter50/run_summary.json, /home/user/图片/gsm8k_improved/autoresearch_scout_cached_teacher_lowaug_iter50/run_summary.json
- research results; baseline_metric 0.485; best_metric 0.485; best_status baseline; best_description Authoritative baseline on 2026-03-27: rerank eval adapter, test[:200], EVAL_USE_CONFIDENCE_RERANK=1, EVAL_NUM_CANDIDATES=8 -> exact_match_rate 0.485 with answer_tag_rate 0.99 and strict_xml_rate 0.98; keep_count 0; latest_status discard  
  status: `supported` | support: `1.0` | evidence: /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/research-results.tsv, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/research-results.tsv, /home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/research-results.tsv
- "experiment_hypothesis": "Replay-heavy retained-adapter continuation may become stable enough for confirmation only when prompt replay rows carry strict teacher trajectories into GRPO and a light teacher-anchor reward nudges outputs back toward those trajectories.",  
  status: `supported` | support: `1.0` | evidence: /home/user/图片/gsm8k_improved/autoresearch_scout_iter102_promptreplay_teacheranchor_keep75c/run_summary.json, /home/user/图片/gsm8k_training_repo/gsm8k_improved/autoresearch_scout_iter102_promptreplay_teacheranchor_keep75c/run_summary.json, /home/user/图片/gsm8k_improved/autoresearch_scout_iter102_promptreplay_teacheranchor_keep75c/run_summary.json

## Critic Verdict

- decision: `continue`
- rationale: The evidence is still narrow or under-verified.
- follow-up: Find more diverse sources that answer: Push the Qwen2.5-0.5B GSM8K confirm200 line beyond 0.565 by prioritizing candidate_distribution and longer-GRPO continuation routes. Do not spend more iterations on rerank_tuning unless a training family beats the baseline. Focus on prompt replay, teacher replay, and longer guarded continuation on percentage and rate_or_ratio.
Current best metric: 0.480 (96/200).
Next candidate proposal label: pair045_neartop_expand.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history. from different communities or toolchains.
- follow-up: What verification and evaluation mechanisms are used in systems addressing: Push the Qwen2.5-0.5B GSM8K confirm200 line beyond 0.565 by prioritizing candidate_distribution and longer-GRPO continuation routes. Do not spend more iterations on rerank_tuning unless a training family beats the baseline. Focus on prompt replay, teacher replay, and longer guarded continuation on percentage and rate_or_ratio.
Current best metric: 0.480 (96/200).
Next candidate proposal label: pair045_neartop_expand.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history.?
- follow-up: How do robust systems persist lessons, failed attempts, or shared knowledge for: Push the Qwen2.5-0.5B GSM8K confirm200 line beyond 0.565 by prioritizing candidate_distribution and longer-GRPO continuation routes. Do not spend more iterations on rerank_tuning unless a training family beats the baseline. Focus on prompt replay, teacher replay, and longer guarded continuation on percentage and rate_or_ratio.
Current best metric: 0.480 (96/200).
Next candidate proposal label: pair045_neartop_expand.
Focus on GSM8K rerank/eval improvements, candidate-pool expansion, selector failure modes, and whether the proposal looks plausible from local experiment history.?

## Lessons

- "prompt_text": "Mode: loop\nContinue improving GSM8K exact match using codex-autoresearch from the current retained best of 0.505 on test[:200]
- Prioritize rerank and inference-time changes in llama3_1_(8b)_grpo.py first; training-side changes are allowed later only if rerank lines stall
- Use mechanical verification only, keep formatting metrics stable, and log every completed experiment before th
- "experiment_risk": "if policy drift is caused by GRPO continuation itself rather than synthetic volume, cutting augmentation from 16 to 8 will not recover the retained verifier-rerank neighborhood",
- research results; baseline_metric 0.485; best_metric 0.485; best_status baseline; best_description Authoritative baseline on 2026-03-27: rerank eval adapter, test[:200], EVAL_USE_CONFIDENCE_RERANK=1, EVAL_NUM_CANDIDATES=8 -> exact_match_rate 0.485 with answer_tag_rate 0.99 and strict_xml_rate 0.98; keep_count 0; latest_status discard
- "experiment_hypothesis": "Replay-heavy retained-adapter continuation may become stable enough for confirmation only when prompt replay rows carry strict teacher trajectories into GRPO and a light teacher-anchor reward nudges outputs back toward those trajectories.",

## Sources

- [gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json](/home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/autoresearch-launch.json)
- [gsm8k_improved/confirm200_hard15_mainline.json](/home/user/图片/gsm8k_improved/confirm200_hard15_mainline.json)
- [gsm8k_improved/autoresearch_scout_iter102_promptreplay_teacheranchor_keep75c/run_summary.json](/home/user/图片/gsm8k_improved/autoresearch_scout_iter102_promptreplay_teacheranchor_keep75c/run_summary.json)
- [gsm8k_improved/autoresearch_scout_iter102_promptreplay_teacheranchor_keep75c/run_summary.json](/home/user/图片/gsm8k_training_repo/gsm8k_improved/autoresearch_scout_iter102_promptreplay_teacheranchor_keep75c/run_summary.json)
- [gsm8k_improved/autoresearch_scout_cached_teacher_lowaug_iter50/run_summary.json](/home/user/图片/gsm8k_improved/autoresearch_scout_cached_teacher_lowaug_iter50/run_summary.json)
- [gsm8k_improved/autoresearch_scout_cached_teacher_lowaug_iter50/run_summary.json](/home/user/图片/gsm8k_training_repo/gsm8k_improved/autoresearch_scout_cached_teacher_lowaug_iter50/run_summary.json)
- [gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/research-results.tsv](/home/user/图片/gsm8k_improved/autoresearch_restore_backups/20260328T092614Z/research-results.tsv)
- [gsm8k_improved/autoresearch_scout_iter97_teacher_anchor_keep75c_rerun/run_summary.json](/home/user/图片/gsm8k_improved/autoresearch_scout_iter97_teacher_anchor_keep75c_rerun/run_summary.json)
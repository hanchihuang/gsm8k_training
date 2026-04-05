# Research Report

## Query

找出当前 GSM8K GRPO 主线里最值得保留的改动、最明显的瓶颈，以及下一步最值得验证的方向

## Plan

- What sub-problems must be solved to answer: 找出当前 GSM8K GRPO 主线里最值得保留的改动、最明显的瓶颈，以及下一步最值得验证的方向?
- What evidence would make the answer to '找出当前 GSM8K GRPO 主线里最值得保留的改动、最明显的瓶颈，以及下一步最值得验证的方向' trustworthy?
- What are the strongest design patterns related to: 找出当前 GSM8K GRPO 主线里最值得保留的改动、最明显的瓶颈，以及下一步最值得验证的方向?
- What failure modes or blind spots appear in systems for: 找出当前 GSM8K GRPO 主线里最值得保留的改动、最明显的瓶颈，以及下一步最值得验证的方向?

## Verified Claims

- lessons summary; recent_lessons L-6: pivot from objective-stability implementation tweaks back to train.toml-level optimizer scheduler and eval-noise control | L-7: pivot away from max_steps sweeps because both 16 and 24 steps regressed against the retained 20-step baseline while rati | L-8: Run summary: 提高 autoresearch-rl-bg 的真实 GSM8K eval_score；起点使用 legacy autoresearch-rl 风格超参基线并保留当前 MathRubric boxed-answer reward / parser 增强。  
  status: `supported` | support: `1.0` | evidence: /home/user/autoresearch-rl-bg/autoresearch-lessons.md, /home/user/autoresearch-rl-bg/autoresearch-lessons.md, /home/user/autoresearch-rl-bg/autoresearch-lessons.md
- autoresearch state; iteration 15; best_metric 0.25; current_metric 0.25; best_iteration 3; last_status pivot; recommended_action needs_human; reason Three strategic pivots were recorded without a keep  
  status: `supported` | support: `1.0` | evidence: /home/user/autoresearch-rl-bg/autoresearch-state.json, /home/user/autoresearch-rl-bg/autoresearch-state.json, /home/user/autoresearch-rl-bg/autoresearch-state.json
- Further unattended relaunches would likely waste effort; the run needs human review, broader scope, or a better metric.; updated_at 2026-04-01T14:31:19Z  
  status: `supported` | support: `1.0` | evidence: /home/user/autoresearch-rl-bg/autoresearch-state.json, /home/user/autoresearch-rl-bg/autoresearch-state.json, /home/user/autoresearch-rl-bg/autoresearch-state.json
- Built on [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), honestly my favourite RL post-training framework out there, and [verifiers](https://github.com/PrimeIntellect-ai/verifiers) for reward verification.  
  status: `supported` | support: `1.0` | evidence: /home/user/autoresearch-rl-bg/README.md, /home/user/autoresearch-rl-bg/README.md, /home/user/autoresearch-rl-bg/README.md
- prime-rl/GRPO-style rollout training, then evaluates locally on GSM8K.  
  status: `supported` | support: `1.0` | evidence: /home/user/autoresearch-rl-bg/run.py, /home/user/autoresearch-rl-bg/run.py, /home/user/autoresearch-rl-bg/run.py
- You can extract the key metric from the log file:  
  status: `supported` | support: `1.0` | evidence: /home/user/autoresearch-rl-bg/program.md, /home/user/autoresearch-rl-bg/program.md, /home/user/autoresearch-rl-bg/program.md

## Critic Verdict

- decision: `continue`
- rationale: The evidence is still narrow or under-verified.
- follow-up: Find more diverse sources that answer: 找出当前 GSM8K GRPO 主线里最值得保留的改动、最明显的瓶颈，以及下一步最值得验证的方向 from different communities or toolchains.
- follow-up: What verification and evaluation mechanisms are used in systems addressing: 找出当前 GSM8K GRPO 主线里最值得保留的改动、最明显的瓶颈，以及下一步最值得验证的方向?
- follow-up: How do robust systems persist lessons, failed attempts, or shared knowledge for: 找出当前 GSM8K GRPO 主线里最值得保留的改动、最明显的瓶颈，以及下一步最值得验证的方向?

## Lessons

- lessons summary; recent_lessons L-6: pivot from objective-stability implementation tweaks back to train.toml-level optimizer scheduler and eval-noise control | L-7: pivot away from max_steps sweeps because both 16 and 24 steps regressed against the retained 20-step baseline while rati | L-8: Run summary: 提高 autoresearch-rl-bg 的真实 GSM8K eval_score；起点使用 legacy autoresearch-rl 风格超参基线并保留当前 MathRubric boxed-answer reward / parser 增强。
- autoresearch state; iteration 15; best_metric 0.25; current_metric 0.25; best_iteration 3; last_status pivot; recommended_action needs_human; reason Three strategic pivots were recorded without a keep
- Further unattended relaunches would likely waste effort; the run needs human review, broader scope, or a better metric.; updated_at 2026-04-01T14:31:19Z
- Built on [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), honestly my favourite RL post-training framework out there, and [verifiers](https://github.com/PrimeIntellect-ai/verifiers) for reward verification.
- prime-rl/GRPO-style rollout training, then evaluates locally on GSM8K.
- You can extract the key metric from the log file:

## Sources

- [autoresearch-lessons.md](/home/user/autoresearch-rl-bg/autoresearch-lessons.md)
- [autoresearch-state.json](/home/user/autoresearch-rl-bg/autoresearch-state.json)
- [run.py](/home/user/autoresearch-rl-bg/run.py)
- [program.md](/home/user/autoresearch-rl-bg/program.md)
- [README.md](/home/user/autoresearch-rl-bg/README.md)
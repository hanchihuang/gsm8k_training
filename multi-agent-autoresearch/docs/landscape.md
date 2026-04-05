# multi-agent autoresearch landscape

This document captures the exact repositories returned by GitHub search for `multi-agent autoresearch` on 2026-04-04 and the design choices distilled from them.

## Distillation

- The strongest repositories treat `autoresearch` as a loop, not a report.
- The strongest `multi-agent` systems add parallel waves, shared memory, and explicit arbitration.
- The weakest projects stop at "search + summarize + write" and never build verification, recovery, or durable lessons.
- Failure attribution is still rare and should be first-class.

## Repositories

1. https://github.com/Human-Agent-Society/CORAL
2. https://github.com/rock-mind/autoresearch-swarm
3. https://github.com/dean0x/autolab
4. https://github.com/deva-harsha-v/AutoResearch-MultiAgent
5. https://github.com/PavanKAgnihotri/AutoResearchLab_MultiAgentAI
6. https://github.com/harishchaurasia/multi-agent-autoresearch
7. https://github.com/hanchihuang/multi-agent-autoresearch
8. https://github.com/FraidoonOmarzai/AutoResearcher
9. https://github.com/devadharshan11-design/AutoResearcher
10. https://github.com/AtlasMindAI/AutoLab
11. https://github.com/chrisliu298/multi-autoresearch
12. https://github.com/Tanmay1112004/AutoResearch-AI---Multi-Agent-Autonomous-Research-System
13. https://github.com/christinetyip/autoresearch-at-home-reports
14. https://github.com/AyushKumar-Singh/AutoResearch-AI-Multi-Agent-LLM-Research-Automation-Platform
15. https://github.com/wildhash/autoresearch-lab
16. https://github.com/djk2017-Rocky/RalphHub
17. https://github.com/rayklanderman/CapstoneProject-Autoresearcher
18. https://github.com/manavchouhan115/Autoresearch.ai
19. https://github.com/dimas-timmers/society-autoresearch
20. https://github.com/zhongjiaqi2002/AutoResearch-Agent
21. https://github.com/Omkar0612/AutoResearchBot
22. https://github.com/vikashmehta292511/autoresearch-lab
23. https://github.com/AmanChourasia7/autoresearch-lab
24. https://github.com/rambo-01/failure-attribution-debugger
25. https://github.com/Vikaash-dev/Autoresearch-v2
26. https://github.com/zabarich/social-sim-study
27. https://github.com/keonhee3337-art/sme-diagnostic-ai
28. https://github.com/Techknowmadlabs/cortex-research-suite

## Kept ideas

- `CORAL`: persistent shared knowledge and runtime durability
- `autoresearch-swarm`: concurrent researchers plus shared result store
- `autolab`: steer/judge/evolve separation instead of unstructured loops
- `multi-autoresearch`: wave-based parallelism
- `Autoresearch.ai`: planner, researcher, critic, writer topology
- `failure-attribution-debugger`: root-cause attribution and trace artifacts
- `society-autoresearch`: lesson-sharing across specialists

## Discarded ideas

- UI-first repos with weak artifacts
- paper/report generators with no verification layer
- systems that rely on agent personas without acceptance criteria
- pipelines that cannot explain why a conclusion is trustworthy


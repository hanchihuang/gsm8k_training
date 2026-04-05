# multi-agent-autoresearch

`multi-agent-autoresearch` 是一个本地优先的多智能体研究运行时。我把当前 GitHub 上 `multi-agent autoresearch` 相关仓库里最有价值的部分抽出来，保留真正能落地的核心：

- `planner / researcher / critic` 迭代闭环
- 并行 research wave
- 持久化证据、结论与 lessons
- 先验证、再综合
- 故障归因与可追溯 artifacts

它不是“输一个主题，吐一篇漂亮报告”的演示项目。它更像一个可审计的研究引擎，目标是把“找证据 -> 发现缺口 -> 追问 -> 验证 -> 输出结论”这个流程跑通。

## 为什么这样设计

这个仓库来自对 GitHub 搜索 `multi-agent autoresearch` 的 28 个精确命中仓库做的去粗存精总结，完整列表见 [docs/landscape.md](docs/landscape.md)。

我主要吸收了这些项目里的强项：

- [Human-Agent-Society/CORAL](https://github.com/Human-Agent-Society/CORAL)
  多智能体编排、共享知识、持久化运行时
- [rock-mind/autoresearch-swarm](https://github.com/rock-mind/autoresearch-swarm)
  并行 wave、共享结果存储、自动汇报
- [dean0x/autolab](https://github.com/dean0x/autolab)
  用 judge / steer / evolve 思路替代拍脑袋式 keep/discard
- [chrisliu298/multi-autoresearch](https://github.com/chrisliu298/multi-autoresearch)
  wave 式并行实验和“早上醒来看总结”的工程风格
- [manavchouhan115/Autoresearch.ai](https://github.com/manavchouhan115/Autoresearch.ai)
  planner -> researchers -> critic -> writer 的拓扑
- [rambo-01/failure-attribution-debugger](https://github.com/rambo-01/failure-attribution-debugger)
  把 failure attribution 当成一等公民

## 功能

- 多智能体流水线：`PlannerAgent`、`ResearchAgent`、`CriticAgent`、`VerifierAgent`、`WriterAgent`
- 标准库线程并发执行 research wave
- 证据账本，保留源级别引用
- claim 级验证与 support score
- 输出 `Markdown` 报告、`JSON` 报告、运行 trace
- 支持离线 `mock` 模式
- 支持 `duckduckgo` 在线搜索
- 支持 `localfs` 本地文件搜索，可直接研究你机器上的代码、日志、报告和配置
- 零第三方运行时依赖

## 快速开始

```bash
cd /home/user/图片/multi-agent-autoresearch
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 离线演示

```bash
mar run \
  --query "多智能体 autoresearch 系统最强的共性是什么？" \
  --output-dir outputs/demo \
  --offline
```

### 在线网页研究

```bash
mar run \
  --query "How should a multi-agent autoresearch runtime balance parallelism, memory, and verification?" \
  --output-dir outputs/web \
  --search-provider duckduckgo
```

### 研究本地代码与实验产物

这条命令会直接分析本地 GRPO for GSM8K 主线：

```bash
mar run \
  --query "找出当前 GSM8K GRPO 主线里最值得保留的改动、最明显的瓶颈，以及下一步最值得验证的方向" \
  --output-dir outputs/gsm8k_local \
  --search-provider localfs \
  --local-root "/home/user/图片/llama3_1_(8b)_grpo.py" \
  --local-root /home/user/图片/gsm8k_improved \
  --local-root /home/user/图片/autoresearch-state.prev.json \
  --max-rounds 3 \
  --max-subquestions 5
```

产物会写到输出目录中：

- `report.md`
- `report.json`
- `run_trace.json`

## CLI

```bash
mar run --query "..." --output-dir outputs/run1 --offline
mar run --query "..." --output-dir outputs/run2 --search-provider duckduckgo
mar run --query "..." --output-dir outputs/run3 --search-provider localfs --local-root /path/a --local-root /path/b
mar landscape --output docs/landscape.md
```

## 搜索后端

- `mock`
  内置语料，适合测试和 demo
- `duckduckgo`
  直接走 DuckDuckGo HTML 搜索，无需 API key
- `localfs`
  遍历本地文件并做轻量相关性匹配，适合研究代码、日志、`run_summary.json`、`README` 等本地资产

## 架构

```text
用户问题
  -> PlannerAgent
  -> 并发 ResearchAgent wave
  -> Evidence Ledger
  -> CriticAgent
      -> 如果发现缺口，继续生成 follow-up wave
  -> VerifierAgent
  -> WriterAgent
  -> Markdown / JSON / Trace artifacts
```

核心模块：

- [src/multi_agent_autoresearch/models.py](src/multi_agent_autoresearch/models.py)
- [src/multi_agent_autoresearch/providers.py](src/multi_agent_autoresearch/providers.py)
- [src/multi_agent_autoresearch/agents.py](src/multi_agent_autoresearch/agents.py)
- [src/multi_agent_autoresearch/engine.py](src/multi_agent_autoresearch/engine.py)
- [src/multi_agent_autoresearch/cli.py](src/multi_agent_autoresearch/cli.py)

## 用它研究本地 GRPO for GSM8K 的建议

你当前这条线已经有几个非常好的输入资产：

- 主训练脚本：`/home/user/图片/llama3_1_(8b)_grpo.py`
- 运行状态：`/home/user/图片/autoresearch-state.prev.json`
- 大量历史实验：`/home/user/图片/gsm8k_improved/`

比较合理的用法不是让它直接替你训练，而是先让它做这三类“研究性工作”：

1. 从历史 `run_summary.json` 里抽取稳定有效的方向
2. 从 `run_report.txt` 与 `README.md` 里识别失败模式和代价高但收益低的尝试
3. 围绕主脚本里的环境变量与奖励设计，提出下一批最值得验证的方向

换句话说，它更适合做“实验编排前的研究和复盘层”，而不是替代你的训练脚本本身。

## 测试

```bash
python -m unittest discover -s tests -v
```

## License

MIT

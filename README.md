# Agent Evaluation Harness

通用 Agent **客观评测、横向对比与失败归因框架**。它不是 Agent，也不依赖被评 Agent 的内部实现：统一接入 trace，将最终 patch 应用到原始源码，重新运行官方测试，再计算指标与归因。

- 已支持：`ours`（项目一 SQLite/JSON/JSONL）、`mini-swe-agent`（外部 JSON 轨迹）
- 主指标：官方 pytest 复测，不使用 LLM 打分，不比较 patch 文本相似度
- 归因：8 类硬规则为主，LLM 仅对低置信结果给建议，人工抽检记录一致率
- 输出：Markdown / JSON / CSV 对比报告、真实数据 PNG 图、trace 回放、bad case 重点观测集

> 仓库不预置“评测结果图”或任何横向对比数字。导入两个 Agent 的真实运行轨迹后，`ae report` 才会生成报告和图；测试目录中的 synthetic fixture 只用于单元测试，不会进入报告。

## 1. 架构

```text
项目一 / mini-SWE-agent
          │
          ▼
  TraceAdapter（统一 Pydantic schema）
          │
          ▼
  版本化 JSONL 评测集 ── few-shot 泄露检查
          │
          ▼
  ObjectiveGrader（隔离目录 + 官方 pytest + 超时）
          │
          ▼
  SQLite（run / attribution / audit / watchlist）
          │
          ├─ MetricsCalculator：pass@k、定位、成本、延迟、重试、抖动
          ├─ RuleBasedAttributor：8 类规则归因
          ├─ FailureClusterer：TF-IDF + KMeans
          └─ ReportGenerator / FastAPI / Gradio
```

```text
src/agent_eval/
├── adapters/       # 双 Agent 适配器与注册表
├── api/            # FastAPI 查询、抽检、报告接口
├── attribution/    # 规则、可选 LLM 辅助、聚类
├── dataset/        # QuixBugs 导入、版本指纹、防泄露
├── evaluation/     # patch 安全应用、客观复测、接入服务
├── metrics/        # 客观指标与统计区间
├── reporting/      # 自动报告与真实数据图表
├── storage/        # SQLite 仓储、审计、观测集
├── cli.py          # 统一命令行
└── demo.py         # Gradio trace 回放与报告
```

## 2. 安装与数据复现

要求 Python 3.10+。

```bash
python -m pip install -e ".[dev,demo]"
```

从官方 QuixBugs 目录导入。导入器只读取 buggy 源码、官方测试和测试数据，**不读取或保存 correct patch**：

```bash
ae --dataset data/quixbugs.v1.jsonl dataset import-quixbugs \
  --root /path/to/QuixBugs --version v1
ae --dataset data/quixbugs.v1.jsonl dataset verify
```

本工作区已生成 `data/quixbugs.v1.jsonl`，包含 40 题；文件内容由 Git 管理，SHA-256 指纹会写入每条 run 和报告。`added_at`、来源、版本、源码、测试、辅助文件均在 JSONL 中。

### few-shot 防泄露

```bash
ae --dataset data/quixbugs.v1.jsonl dataset check-leakage \
  --few-shot /path/to/fewshot.txt
```

发现当前题 ID、函数名或文件路径即返回非零退出码。注意：这只检查显式引用，正式实验还应在构造 few-shot 时按 problem_id 做集合级排除。

## 3. 同一脚本评两个 Agent

### 项目一

```bash
ae --dataset data/quixbugs.v1.jsonl ingest \
  --agent ours \
  --source /path/to/coding-agent/var/runs.db \
  --experiment quixbugs-s4-r3
```

适配器直接读取项目一 `runs.trace_json`，展开内部 attempts。内部重试不会冒充独立生成；`repeat_index` 映射为独立生成序号。

### mini-SWE-agent

```bash
ae --dataset data/quixbugs.v1.jsonl ingest \
  --agent mini-swe-agent \
  --source /path/to/mini-swe-agent/trajectories \
  --experiment quixbugs-mini-r3
```

适配器支持常见 `info + messages` JSON、`tool_calls`、bash 代码块及显式 `steps` 结构。`info.submission` 或最后的 `git diff` 作为最终 patch；`submitted` 不会被误判为测试通过。日志没有的 token/latency 记为 0（未知），不虚构。单文件缺少题号时可加 `--problem-id`，重复序号可加 `--attempt-index` 或 `--attempt-offset`。

两条命令都会执行同一流程：**适配 → 应用 patch → 官方测试复测 → SQLite 落库 → 自动归因**。原始日志里的 pass/fail 仅保存在 `metadata.logged_test_result`，不参与最终评分。修改官方测试、目录穿越和非法 patch 均判失败。

## 4. 重复实验与指标口径

同一配置独立运行至少 3 次，并为每次生成保留递增的 `attempt_index`（项目一使用 `repeat_index`）。然后执行：

```bash
ae --dataset data/quixbugs.v1.jsonl report \
  --experiment quixbugs-s4-r3 --agent ours --agent mini-swe-agent --k 3
```

若两个 Agent 的实验 ID 不同，先使用相同实验 ID 导入，或在同一实验批次中分别保存 run。报告自动只保留共同 problem_id，覆盖不齐时写入警告，避免分母偷换。

| 指标 | 口径 |
|---|---|
| pass@1 | 每题最小 `attempt_index` 的独立生成通过比例 |
| pass@k | 每题前 k 次**独立生成**至少一次官方测试全绿；内部重试不计入 k |
| repeat 均值/方差 | 每轮独立生成通过率的均值与样本方差；单次时方差为 N/A |
| 95% 区间 | 全部独立 run 成功比例的 Wilson 区间 |
| 定位 Top-1/Top-3 | 首次/前 3 次 read 类动作是否触达已知 buggy 文件；无读取计失败 |
| token/run | 所有 run 的总 token ÷ run 数 |
| token/problem | 所有 run 的总 token ÷ 唯一题数 |
| P50/P95 | 步骤延迟分位数；适配器以 0 表示未知，未知值排除 |
| 重试成功率 | `retry_count > 0` 的 run 中，最终官方测试通过比例 |
| 错误分布 | 仅失败 run，优先使用已保存的归因类别 |

同一配置多次运行取均值并报告方差，用于对抗 LLM 随机性；`temperature=0` 也不能替代重复实验。

## 5. 归因规则与人工抽检

规则优先级如下，高优先级命中后停止：

1. `environment`：timeout、OOM、依赖缺失；
2. `syntax_error`：SyntaxError、IndentationError 等；
3. `loop_exhausted`：相同 action + 规范化 args 连续至少 3 次，或明确循环/步数耗尽；
4. `tool_failure`：工具失败阻断有效 patch，或 patch 无法应用；
5. `wrong_location`：改动文件与 gold 文件完全不匹配；
6. `regression`：历史失败消失，同时出现此前没有的新失败测试；
7. `assertion_mismatch`：patch 命中正确文件，但最终仍有断言不一致；
8. `incomplete_fix`：以上均不能确定的剩余失败，低置信并强制人工复核。

归因同时读取测试输出、patch 改动文件和 trace 动作序列，不是“让 LLM 看一眼”。每条归因保存规则证据、置信度、方法和是否需复核。

### 可选 LLM 辅助

复制 `.env.example` 为 `.env` 并配置 OpenAI 兼容接口，然后：

```bash
ae --dataset data/quixbugs.v1.jsonl attribute --experiment quixbugs-s4-r3 --llm
```

LLM 只处理低置信 case，只能输出限定的 8 类 JSON；网络或格式失败回退规则结果。它不能修改测试结果，且输出仍标记 `needs_review=true`。

人工抽检：

```bash
ae audit record --run-id RUN_ID --reviewer NAME \
  --category incomplete_fix --note "核对 trace 与官方测试"
ae audit agreement
```

未抽检时一致率为 `null`，报告展示 N/A，不会自动填数字。

## 6. bad case 回流闭环

```bash
ae watchlist refresh --min-failures 2
ae watchlist list
ae --dataset data/quixbugs.v1.jsonl watchlist export \
  --output data/priority_problems.json
```

观测集从当前 SQLite 的失败 run 重新聚合，而不是手工维护名单。`priority_problems.json` 将高频失败题排在最前，并附带数据集指纹。项目一 `ca eval run` 支持 `--priority path/to/priority_problems.json`，会按此顺序优先执行这些题，从而形成“评测归因 → 重点题优先复跑 → 验证修复”的闭环。只有真实前后两轮报告才能声明提升，框架不会自动生成“提升了多少”的叙事。

## 7. 报告、API 与 demo

报告目录包含：

- `report.md`：指标、归因分布、自动差异结论、Top-3 聚类、典型 bad case 与 trace 链接、抽检一致率；
- `metrics.json` / `metrics.csv`：完整可复算数据；
- `pass_at_1.png`、`error_distribution.png`、`token_success.png`：全部由当前 run 绘制；
- `traces/*.json`：bad case 原始统一轨迹。

启动服务：

```bash
ae --dataset data/quixbugs.v1.jsonl serve --host 0.0.0.0 --port 8000
ae demo --server-name 0.0.0.0 --port 7860
```

FastAPI 提供 `/health`、`/runs`、`/runs/{run_id}`、`/runs/{run_id}/regrade`、`/watchlist`、`/watchlist/refresh`、`/runs/{run_id}/audit`、`/audits/agreement`、`/reports`。复测接口使用当前版本数据集重新执行官方测试，并替换上一次客观复测记录，不会重复追加历史。Gradio 支持输入 run_id 回放每一步、查看归因和最终 patch，并展示最近一次真实报告。

Hugging Face Spaces 使用 Gradio SDK，`app.py` 为入口，`requirements.txt` 为依赖。部署前将**真实评测数据库**放到 Space 可读的持久路径，并设置 `AGENT_EVAL_DB_PATH`；空数据库不会生成演示数字或示意图。README 第一屏的对比截图也应在真实双 Agent 报告生成后替换，不提前放占位图。

## 8. 验证

```bash
python -m pytest
python -m ruff check src tests app.py
python -m compileall -q src app.py
```

测试覆盖双适配器、8 类归因、指标统计、patch 安全、官方测试复测、日志伪通过纠偏、SQLite 审计与回流、报告和 API。所有测试数据均位于临时目录，不写入正式报告。

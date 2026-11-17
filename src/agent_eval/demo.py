from __future__ import annotations

import html
import json
from pathlib import Path

from agent_eval.config import Settings
from agent_eval.storage import EvaluationRepository


def create_demo(settings: Settings | None = None):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio 未安装，请执行 pip install -e '.[demo]'") from exc

    settings = settings or Settings()
    repository = EvaluationRepository(settings.db_path)
    reports_root = Path(settings.reports_dir)

    def replay(run_id: str):
        trace = repository.get_run(run_id.strip())
        if trace is None:
            return "未找到 run_id。", "{}", ""
        attribution = repository.get_attribution(trace.run_id)
        lines = [
            f"### {trace.agent_name} / {trace.problem_id}",
            "",
            f"- 实验：`{trace.experiment_id}`；独立生成序号：{trace.attempt_index}",
            f"- 客观复测：**{trace.test_result.value}**；策略：`{trace.strategy or '-'}`",
            f"- token：{trace.total_tokens}；步骤：{len(trace.steps)}；"
            f"内部重试：{trace.retry_count}",
            "",
            "### Trace 回放",
            "",
        ]
        for step in trace.steps:
            arguments = html.escape(json.dumps(step.args, ensure_ascii=False))
            observation = html.escape(step.observation)
            thought = html.escape(step.thought)
            lines.extend(
                [
                    f"<details><summary>Step {step.step_id} · {html.escape(step.action)} · "
                    f"{step.latency_ms} ms · {step.tokens} tokens · ok={step.ok}</summary>",
                    f"<p><b>Thought</b>: {thought}</p>",
                    f"<pre>{arguments}</pre>",
                    f"<pre>{observation}</pre></details>",
                    "",
                ]
            )
        lines.extend(["### 最终官方测试输出", "", "```text", trace.test_output, "```"])
        attribution_json = (
            attribution.model_dump_json(indent=2) if attribution else '{"status": "pass 或未归因"}'
        )
        return "\n".join(lines), attribution_json, trace.final_patch

    def latest_report():
        pointer_path = reports_root / "latest.json"
        if not pointer_path.exists():
            return "尚未生成报告。请先运行 ae report。", None, None, None
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        report_dir = Path(pointer["report_dir"])
        markdown = (report_dir / "report.md").read_text(encoding="utf-8")
        return (
            markdown,
            str(report_dir / "pass_at_1.png"),
            str(report_dir / "error_distribution.png"),
            str(report_dir / "token_success.png"),
        )

    with gr.Blocks(title="Agent 评测与失败归因") as demo:
        gr.Markdown(
            "# Agent 评测与失败归因\n"
            "主指标来自官方测试复测；规则归因与可选 LLM 辅助建议分开记录。"
        )
        with gr.Tab("Trace 回放"):
            run_id = gr.Textbox(label="run_id", placeholder="输入数据库中的 run_id")
            button = gr.Button("加载轨迹", variant="primary")
            trace_output = gr.Markdown()
            attribution_output = gr.Code(language="json", label="归因结果")
            patch_output = gr.Textbox(label="最终 unified diff", lines=14)
            button.click(
                replay,
                inputs=run_id,
                outputs=[trace_output, attribution_output, patch_output],
            )
        with gr.Tab("最新对比报告"):
            refresh = gr.Button("刷新报告")
            report_output = gr.Markdown()
            pass_image = gr.Image(label="pass@1")
            error_image = gr.Image(label="错误类别分布")
            token_image = gr.Image(label="Token 与成功率")
            refresh.click(
                latest_report,
                outputs=[report_output, pass_image, error_image, token_image],
            )
            demo.load(
                latest_report,
                outputs=[report_output, pass_image, error_image, token_image],
            )
    return demo

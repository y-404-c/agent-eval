"""薄适配：把 QuixBugs 单文件题喂给 mini-SWE-agent，产出轨迹 JSON。

为每道题建一个**临时 git 仓库**，放入 buggy 源码 + 官方测试（官方测试
绝不修改，项目二的 grader 会拒绝改测试的 patch），取函数 docstring 规格
作为 issue 描述，在本地 Git Bash 环境里跑 mini-SWE-agent（v1），把轨迹
写入固定目录。项目二的 MiniSweAgentAdapter 随后读取这些轨迹并做官方
测试客观复测。

设计约束：
- 本脚本只依赖标准库 + mini-swe-agent（装在独立的 ``.venv-mini`` 里），
  不 import ``agent_eval``，避免污染项目二的运行环境，也不引入版本冲突。
- 环境优先用 Git Bash（Windows 上没有 Docker / WSL 时的可行方案），
  保证 agent 发出的 ``cat`` / ``sed`` / ``git`` / ``pytest`` 都能执行。
- 提供 ``--deterministic`` 模式：用内置固定输出零成本跑通全链路
  （建仓库 → agent → 存轨迹 → 项目二解析），不消耗 API 额度。
"""

from __future__ import annotations

# 在 import minisweagent 之前关闭启动横幅，并让 litellm 成本统计容错。
import os

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "quixbugs.v1.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "trajectories" / "mini-swe-agent"

# 零成本验证用的固定动作序列：read → 跑失败测试 → 修 → 跑通过测试 → 提交。
# 对应 bitcount 题：把 ``n ^= n - 1`` 改成 ``n &= n - 1``。
DETERMINISTIC_OUTPUTS: list[str] = [
    "THOUGHT: 先查看源码定位 bug。\n```bash\ncat python_programs/bitcount.py\n```",
    "THOUGHT: 运行官方测试确认当前失败。\n```bash\npytest python_testcases/test_bitcount.py -q\n```",
    "THOUGHT: bitcount 应为 n &= n - 1，修复它。\n```bash\nsed -i 's/n ^= n - 1/n \\&= n - 1/' python_programs/bitcount.py\n```",
    "THOUGHT: 修复后重跑测试确认通过。\n```bash\npytest python_testcases/test_bitcount.py -q\n```",
    "THOUGHT: 完成，提交最终 diff。\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached\n```",
]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def find_git_bash() -> Path:
    candidates = [
        os.getenv("MSWEA_GIT_BASH"),
        r"D:\Git\Git\bin\bash.exe",
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    which = shutil.which("bash")
    if which:
        return Path(which)
    raise RuntimeError(
        "找不到 bash.exe。请安装 Git for Windows，或设置环境变量 MSWEA_GIT_BASH 指向 bash.exe。"
    )


def venv_scripts_dir() -> Path:
    return Path(sys.executable).resolve().parent


def load_problems(dataset_path: Path) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def extract_spec(source_code: str) -> str:
    """取源码末尾的 docstring 规格（QuixBugs 每题的自然语言规格）。"""
    blocks = re.findall(r"(" + '"""' + r"|'''|)(.*?)\1", source_code, re.DOTALL)
    if blocks:
        return blocks[-1][1].strip()
    return source_code.strip()


def materialize_repo(problem: dict[str, Any], root: Path) -> None:
    """把 buggy 源码 + 官方测试 + 支撑文件物化为临时 git 仓库并做初始提交。"""
    supports: dict[str, str] = problem.get("support_files") or {}
    for rel_path, content in supports.items():
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    entry_path = Path(problem["entry_path"])
    (root / entry_path).parent.mkdir(parents=True, exist_ok=True)
    (root / entry_path).write_text(problem["source_code"], encoding="utf-8")

    # 让 python_programs 成为可导入包，测试里的 `from python_programs.X import Y` 才能工作。
    pkg_init = root / entry_path.parent / "__init__.py"
    if not pkg_init.exists():
        pkg_init.write_text("", encoding="utf-8")

    test_path = Path(problem["test_path"])
    (root / test_path).parent.mkdir(parents=True, exist_ok=True)
    (root / test_path).write_text(problem["test_code"], encoding="utf-8")

    # 防止 pytest 生成的字节码/缓存被 git add -A 带进最终 patch。
    (root / ".gitignore").write_text(
        "__pycache__/\n*.pyc\n*.pyo\n.pytest_cache/\n", encoding="utf-8"
    )

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "agent-eval@local"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "agent-eval"], cwd=root, check=True)
    # 关闭换行符转换，避免 git diff 输出 CRLF 警告污染最终 patch。
    subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=root, check=True)
    subprocess.run(["git", "config", "core.safecrlf", "false"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial buggy snapshot"], cwd=root, check=True)


class BashEnvironment:
    """mini-SWE-agent Environment 的 Git Bash 实现（Windows 无 Docker/WSL 时使用）。"""

    def __init__(
        self,
        repo: Path,
        bash: Path,
        venv_scripts: Path,
        timeout: int = 60,
    ) -> None:
        self.repo = repo
        self.bash = bash
        self.timeout = timeout
        self.config = {
            "cwd": str(repo),
            "bash": str(bash),
            "timeout": timeout,
            "shell": "git-bash",
        }
        self._env = self._build_env(venv_scripts)

    def _build_env(self, venv_scripts: Path) -> dict[str, str]:
        env = os.environ.copy()
        prepend = os.pathsep.join(
            [
                str(venv_scripts),
                str(self.bash.parent),
                str(self.bash.parent.parent / "usr" / "bin"),
            ]
        )
        env["PATH"] = prepend + os.pathsep + env.get("PATH", "")
        env["PYTHONPATH"] = str(self.repo) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        env.update(
            {
                "PAGER": "cat",
                "PIP_PROGRESS_BAR": "off",
                "TQDM_DISABLE": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        return env

    def execute(self, command: str, cwd: str = "") -> dict[str, str]:
        work = cwd or str(self.repo)
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "stdout.log"
            err_path = Path(tmp) / "stderr.log"
            with out_path.open("w", encoding="utf-8") as out_fh, err_path.open(
                "w", encoding="utf-8"
            ) as err_fh:
                proc = subprocess.Popen(
                    [str(self.bash), "-lc", command],
                    cwd=work,
                    env=self._env,
                    stdout=out_fh,
                    stderr=err_fh,
                    creationflags=creationflags,
                )
                try:
                    proc.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    # 杀整个进程树：Windows 上只杀 bash 会留下孙子进程
                    # （如无限循环的 pytest），且其持有的管道会让父进程死锁。
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True,
                    )
                    proc.wait()
                    output = _read_text(out_path) + _read_text(err_path)
                    raise subprocess.TimeoutExpired(
                        command,
                        self.timeout,
                        output=output.encode("utf-8", "replace"),
                    ) from None
            output = _read_text(out_path) + _read_text(err_path)
            # returncode 必须是 int：mini-SWE-agent 的 has_finished 用
            # ``output.get("returncode", 0) != 0`` 判断提交是否成功，
            # 字符串 "0" 会被误判为失败，导致 agent 反复重发提交命令。
            return {"output": output, "returncode": proc.returncode}

    def get_template_vars(self) -> dict[str, Any]:
        return {}


def load_swebench_agent_config() -> dict[str, Any]:
    """复用 mini-SWE-agent 自带的 swebench 提示词模板（bash 代码块格式）。"""
    import yaml  # noqa: WPS433 (deliberately local import)

    from minisweagent import package_dir

    cfg_path = Path(package_dir) / "config" / "extra" / "swebench.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return dict(cfg["agent"])


def make_model(args: argparse.Namespace) -> Any:
    if args.deterministic:
        from minisweagent.models.test_models import DeterministicModel

        return DeterministicModel(outputs=list(DETERMINISTIC_OUTPUTS))

    api_key = (
        args.api_key
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("MSWEA_MODEL_API_KEY")
    )
    if not api_key:
        raise SystemExit(
            "缺少 API key：请传 --api-key，或设置环境变量 "
            "DEEPSEEK_API_KEY / OPENAI_API_KEY / MSWEA_MODEL_API_KEY。"
        )

    model_name = args.model
    if "/" not in model_name:
        model_name = f"openai/{model_name}"

    from minisweagent.models import get_model

    config = {
        "model_class": "litellm",
        "model_kwargs": {
            "api_base": args.base_url,
            "api_key": api_key,
            "temperature": args.temperature,
            "drop_params": True,
        },
        "cost_tracking": "ignore_errors",
    }
    return get_model(model_name, config)


def run_one(
    problem: dict[str, Any],
    args: argparse.Namespace,
    agent_cfg: dict[str, Any],
    output_dir: Path,
    bash: Path,
    venv_scripts: Path,
) -> Path:
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.run.utils.save import save_traj

    problem_id = problem["problem_id"]
    out_file = output_dir / f"{problem_id}.traj.json"

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / problem_id
        repo.mkdir()
        materialize_repo(problem, repo)
        env = BashEnvironment(repo, bash, venv_scripts, timeout=args.timeout)
        agent = DefaultAgent(make_model(args), env, **agent_cfg)

        spec = extract_spec(problem["source_code"])
        task = (
            f"Fix the buggy implementation in {problem['entry_path']} so the official tests pass.\n\n"
            f"Specification (from the function docstring):\n{spec}"
        )
        try:
            status, submission = agent.run(task)
        except Exception as exc:  # noqa: BLE001 - 捕获后落盘，避免单题失败中断全量
            status, submission = "RUN_ERROR", repr(exc)

        save_traj(
            agent,
            out_file,
            print_path=False,
            exit_status=status,
            result=submission,
            extra_info={
                "instance_id": problem_id,
                "problem_id": problem_id,
                "entry_path": problem.get("entry_path"),
                "model": args.model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            problem_id=problem_id,
            instance_id=problem_id,
        )
    return out_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run mini-SWE-agent on QuixBugs problems and save trajectories."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--problem-id", action="append", default=None,
                        help="只跑指定题，可多次指定。")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题。")
    parser.add_argument("--deterministic", action="store_true",
                        help="用固定输出零成本跑通链路（不调用 API，仅 bitcount）。")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--step-limit", type=int, default=40)
    parser.add_argument("--cost-limit", type=float, default=5.0)
    parser.add_argument("--timeout", type=int, default=60, help="单条命令超时秒数。")
    parser.add_argument("--force", action="store_true",
                        help="即使输出轨迹已存在也重跑（默认跳过已存在的，断点续跑）。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problems = load_problems(args.dataset)
    if args.problem_id:
        wanted = set(args.problem_id)
        problems = [p for p in problems if p["problem_id"] in wanted]
    if args.limit is not None:
        problems = problems[: args.limit]
    if not problems:
        raise SystemExit("没有可运行的题目。")

    bash = find_git_bash()
    venv_scripts = venv_scripts_dir()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_cfg = load_swebench_agent_config()
    agent_cfg["step_limit"] = args.step_limit
    agent_cfg["cost_limit"] = args.cost_limit

    print(f"[mini-swe-runner] dataset={args.dataset} problems={len(problems)} "
          f"bash={bash} deterministic={args.deterministic}")
    for index, problem in enumerate(problems, start=1):
        pid = problem["problem_id"]
        existing = output_dir / f"{pid}.traj.json"
        if existing.exists() and not args.force:
            print(f"[{index}/{len(problems)}] {pid}: SKIP (exists) -> {existing}")
            continue
        try:
            path = run_one(problem, args, agent_cfg, output_dir, bash, venv_scripts)
            print(f"[{index}/{len(problems)}] {pid}: OK -> {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[{index}/{len(problems)}] {pid}: FAILED -> {exc!r}", file=sys.stderr)


if __name__ == "__main__":
    main()

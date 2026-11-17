from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import PurePosixPath

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class PatchError(RuntimeError):
    pass


def changed_files(patch: str) -> list[str]:
    paths: list[str] = []
    for line in patch.splitlines():
        if not line.startswith("+++ "):
            continue
        path = _clean_path(line[4:].split("\t", 1)[0])
        if path and path != "/dev/null":
            paths.append(path)
    return list(dict.fromkeys(paths))


def apply_unified_diff(patch: str, files: Mapping[str, str]) -> dict[str, str]:
    """应用文本 unified diff；拒绝绝对路径、目录穿越与非法上下文。"""

    result = dict(files)
    lines = patch.splitlines()
    cursor = 0
    found = False
    while cursor < len(lines):
        if not lines[cursor].startswith("--- "):
            cursor += 1
            continue
        if cursor + 1 >= len(lines) or not lines[cursor + 1].startswith("+++ "):
            raise PatchError("patch 头部缺少 +++ 行")
        old_path = _clean_path(lines[cursor][4:].split("\t", 1)[0])
        new_path = _clean_path(lines[cursor + 1][4:].split("\t", 1)[0])
        path = new_path if new_path != "/dev/null" else old_path
        _validate_path(path)
        cursor += 2
        hunks: list[tuple[tuple[str, ...], list[str]]] = []
        while cursor < len(lines) and lines[cursor].startswith("@@"):
            match = _HUNK_RE.match(lines[cursor])
            if not match:
                raise PatchError(f"无法解析 hunk: {lines[cursor]}")
            cursor += 1
            body: list[str] = []
            while cursor < len(lines) and (
                lines[cursor].startswith((" ", "+", "-", "\\")) or lines[cursor] == ""
            ):
                body.append(lines[cursor])
                cursor += 1
            hunks.append((match.groups(), body))
        if not hunks:
            raise PatchError(f"文件 {path} 没有 hunk")
        source = result.get(path, "") if old_path == "/dev/null" else result.get(path)
        if source is None:
            raise PatchError(f"patch 指向不存在的文件: {path}")
        result[path] = _apply_hunks(source, hunks)
        found = True
    if patch.strip() and not found:
        raise PatchError("不是可识别的 unified diff")
    return result


def _apply_hunks(source: str, hunks: list[tuple[tuple[str, ...], list[str]]]) -> str:
    source_lines = source.splitlines()
    output: list[str] = []
    position = 0
    for groups, body in hunks:
        old_start, old_count = int(groups[0]), int(groups[1] or "1")
        start = old_start - 1 if old_count else old_start
        if start < position or start > len(source_lines):
            raise PatchError("hunk 位置越界或重叠")
        output.extend(source_lines[position:start])
        position = start
        for raw in body:
            tag, text = (raw[0], raw[1:]) if raw else (" ", "")
            if tag == " ":
                if position >= len(source_lines) or source_lines[position] != text:
                    raise PatchError(f"上下文不匹配: {text!r}")
                output.append(source_lines[position])
                position += 1
            elif tag == "-":
                if position >= len(source_lines) or source_lines[position] != text:
                    raise PatchError(f"删除行不匹配: {text!r}")
                position += 1
            elif tag == "+":
                output.append(text)
            elif tag != "\\":
                raise PatchError(f"非法 diff 行: {raw!r}")
    output.extend(source_lines[position:])
    return "\n".join(output) + ("\n" if source.endswith("\n") or output else "")


def _clean_path(path: str) -> str:
    path = path.strip().strip('"').replace("\\", "/")
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path


def _validate_path(path: str) -> None:
    pure = PurePosixPath(path)
    if not path or path == "/dev/null" or pure.is_absolute() or ".." in pure.parts:
        raise PatchError(f"不安全或无效的 patch 路径: {path!r}")
    if re.match(r"^[A-Za-z]:", path):
        raise PatchError(f"不允许绝对路径: {path}")

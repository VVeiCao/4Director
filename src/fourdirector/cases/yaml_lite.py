"""Small YAML subset loader so Case configs need no extra dependency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        payload = json.loads(text)
    else:
        payload = _parse(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def _parse(text: str) -> Any:
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, raw.strip()))
    value, index = _parse_block(lines, 0, 0)
    if index != len(lines):
        raise ValueError("YAML subset parser did not consume the document")
    return value


def _parse_block(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index
    current_indent, content = lines[index]
    if current_indent < indent:
        return {}, index
    if content.startswith("- "):
        return _parse_list(lines, index, current_indent)
    return _parse_dict(lines, index, current_indent)


def _parse_dict(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[dict[str, Any], int]:
    payload: dict[str, Any] = {}
    while index < len(lines):
        current_indent, content = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"invalid YAML indent: {content}")
        if content.startswith("- "):
            raise ValueError(f"expected mapping entry, got {content}")
        if ":" not in content:
            raise ValueError(f"expected key: value, got {content}")
        key, remainder = content.split(":", 1)
        key = key.strip()
        remainder = remainder.strip()
        index += 1
        if remainder:
            payload[key] = _parse_scalar(remainder)
            continue
        nested, index = _parse_block(lines, index, indent + 2)
        payload[key] = nested
    return payload, index


def _parse_list(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[list[Any], int]:
    items: list[Any] = []
    while index < len(lines):
        current_indent, content = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent or not content.startswith("- "):
            raise ValueError(f"invalid YAML list item: {content}")
        remainder = content[2:].strip()
        index += 1
        if remainder:
            if remainder.endswith(":") and ":" in remainder[:-1]:
                raise ValueError(f"unsupported inline nested mapping: {remainder}")
            if ":" in remainder and not remainder.startswith(("{", "[", "'", '"')):
                key, value = remainder.split(":", 1)
                item = {key.strip(): _parse_scalar(value.strip())}
                nested, index = _parse_block(lines, index, indent + 2)
                if isinstance(nested, dict):
                    item.update(nested)
                items.append(item)
            else:
                items.append(_parse_scalar(remainder))
            continue
        nested, index = _parse_block(lines, index, indent + 2)
        items.append(nested)
    return items, index


def _parse_scalar(value: str) -> Any:
    if value in {"", "null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]
    try:
        if value.startswith("0") and value not in {"0"} and not value.startswith("0."):
            return value
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value

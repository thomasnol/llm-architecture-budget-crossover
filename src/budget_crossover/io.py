from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def read_jsonl(path: Path, model: type[T]) -> list[T]:
    if not path.exists():
        return []
    rows: list[T] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(model.model_validate_json(line))
    return rows


def append_jsonl(path: Path, row: BaseModel | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(path)

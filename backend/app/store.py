"""Bounded in-memory workspace for datasets and chat sessions.

This is deliberately simple: a single-process dictionary guarded by a lock. It keeps the
project easy to run anywhere (no database required). Swap it for Redis / a database
before running more than one worker.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import pandas as pd

from .analysis import coerce_types
from .demo import DEMO_NAME, build_demo_dataframe

DEMO_ID = "demo"


@dataclass
class ChatTurn:
    role: str  # "user" | "assistant"
    content: str
    charts: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class Dataset:
    id: str
    name: str
    df: pd.DataFrame
    source: str  # "demo" | "upload"
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    sessions: dict[str, list[ChatTurn]] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_used_at = time.time()

    def history(self, session_id: str) -> list[ChatTurn]:
        return self.sessions.setdefault(session_id, [])


class DatasetStore:
    def __init__(self, max_datasets: int = 6) -> None:
        self._max = max_datasets
        self._lock = threading.RLock()
        self._items: OrderedDict[str, Dataset] = OrderedDict()

    # ── datasets ───────────────────────────────────────────────────────────
    def ensure_demo(self) -> Dataset:
        with self._lock:
            if DEMO_ID not in self._items:
                ds = Dataset(
                    id=DEMO_ID,
                    name=DEMO_NAME,
                    df=coerce_types(build_demo_dataframe()),
                    source="demo",
                )
                self._items[DEMO_ID] = ds
                self._items.move_to_end(DEMO_ID, last=False)
            return self._items[DEMO_ID]

    def add(self, name: str, df: pd.DataFrame) -> Dataset:
        with self._lock:
            uploads = [k for k, v in self._items.items() if v.source == "upload"]
            # Evict least-recently-used uploads; the demo dataset is never evicted.
            while len(uploads) >= self._max:
                oldest = min(uploads, key=lambda k: self._items[k].last_used_at)
                self._items.pop(oldest, None)
                uploads.remove(oldest)
            ds = Dataset(id=uuid.uuid4().hex[:12], name=name, df=df, source="upload")
            self._items[ds.id] = ds
            return ds

    def get(self, dataset_id: str) -> Dataset | None:
        with self._lock:
            ds = self._items.get(dataset_id)
            if ds is None and dataset_id == DEMO_ID:
                ds = self.ensure_demo()
            if ds is not None:
                ds.touch()
            return ds

    def delete(self, dataset_id: str) -> bool:
        with self._lock:
            if dataset_id == DEMO_ID:
                return False
            return self._items.pop(dataset_id, None) is not None

    def list(self) -> list[Dataset]:
        with self._lock:
            self.ensure_demo()
            return sorted(self._items.values(), key=lambda d: (d.source != "demo", -d.created_at))

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

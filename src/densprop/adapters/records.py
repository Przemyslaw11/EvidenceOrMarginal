"""RecordStore on a directory: CSV (also .csv.gz), JSON and text files by name relative to it."""

from __future__ import annotations

import glob
import json
import os
from typing import Any

import pandas as pd


class FileRecords:
    def __init__(self, root: str):
        self.root = str(root)

    def path(self, name: str) -> str:
        return os.path.join(self.root, name)

    def exists(self, name: str) -> bool:
        return os.path.exists(self.path(name))

    def names(self, pattern: str) -> list[str]:
        return sorted(os.path.relpath(p, self.root) for p in glob.glob(self.path(pattern)))

    def read_csv(self, name: str, **kw: Any) -> pd.DataFrame:
        return pd.read_csv(self.path(name), **kw)

    def read_json(self, name: str) -> Any:
        with open(self.path(name)) as f:
            return json.load(f)

    def read_text(self, name: str) -> str:
        with open(self.path(name)) as f:
            return f.read()

    def _parent(self, name: str) -> str:
        p = self.path(name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def write_csv(self, name: str, df: pd.DataFrame, **kw: Any) -> None:
        """Without the index unless asked (`index=True`); other to_csv options pass through."""
        df.to_csv(self._parent(name), **{"index": False, **kw})

    def write_json(self, name: str, obj: Any, **kw: Any) -> None:
        with open(self._parent(name), "w") as f:
            json.dump(obj, f, **kw)

    def write_text(self, name: str, text: str) -> None:
        with open(self._parent(name), "w") as f:
            f.write(text)

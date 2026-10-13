from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass
class MimicPaths:
    root: str = "mimic-iv-3.1"
    gz: bool = True   # real tables are .csv.gz; fixtures are plain .csv

    def _ext(self) -> str:
        return ".csv.gz" if self.gz else ".csv"

    def icu(self, name: str) -> str:
        sub = "icu" if self.gz else ""
        return os.path.join(self.root, sub, name + self._ext())

    def hosp(self, name: str) -> str:
        sub = "hosp" if self.gz else ""
        return os.path.join(self.root, sub, name + self._ext())

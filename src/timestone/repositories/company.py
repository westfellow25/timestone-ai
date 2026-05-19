"""Loads and saves Company digital twins."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from ..domain.company import Company
from ..infrastructure.paths import REPO_ROOT, TWINS_DIR


class CompanyRepository:
    """Loads twin files from data/twins/ and (for backward compat)
    *_twin.json files in repo root."""

    def __init__(self, twins_dir: Optional[Path] = None, also_repo_root: bool = True):
        self.twins_dir = Path(twins_dir) if twins_dir else TWINS_DIR
        self.also_repo_root = also_repo_root

    def list_all(self) -> List[Company]:
        companies = []
        # New location
        if self.twins_dir.exists():
            for p in sorted(self.twins_dir.glob("*.json")):
                companies.append(Company.from_dict(json.loads(p.read_text())))
        # Legacy location (repo root)
        if self.also_repo_root:
            for p in sorted(REPO_ROOT.glob("*_twin.json")):
                companies.append(Company.from_dict(json.loads(p.read_text())))
            for p in sorted(REPO_ROOT.glob("*_digital_twin.json")):
                companies.append(Company.from_dict(json.loads(p.read_text())))
        # Deduplicate by company_name (first wins)
        seen, out = set(), []
        for c in companies:
            if c.company_name in seen:
                continue
            seen.add(c.company_name)
            out.append(c)
        return out

    def load_by_name(self, name: str) -> Optional[Company]:
        for c in self.list_all():
            if c.company_name.lower() == name.lower():
                return c
        return None

    def save(self, company: Company) -> Path:
        self.twins_dir.mkdir(parents=True, exist_ok=True)
        safe = company.company_name.lower().replace(" ", "_").replace("/", "_")
        path = self.twins_dir / f"{safe}.json"
        path.write_text(json.dumps(company.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

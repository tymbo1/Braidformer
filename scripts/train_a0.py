#!/usr/bin/env python3
"""Run or validate an explicitly approved bounded BraidNano A0/A1 pilot."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from braidformer.training.runner import main


if __name__ == "__main__":
    raise SystemExit(main())

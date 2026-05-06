#!/usr/bin/env python3
"""Smoke-test that ``denoise_dataset.py`` imports cleanly (matches user-site stripping there).

Run from repo root with the target conda env active, e.g.:

  conda activate refining
  python refiner_data_generation/verify_imports.py
"""

from __future__ import annotations

import importlib.util
import os
import site
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))

    user_site = site.getusersitepackages()
    nu = user_site.rstrip(os.sep)
    sys.path[:] = [p for p in sys.path if p.rstrip(os.sep) != nu]

    target = Path(__file__).resolve().parent / "denoise_dataset.py"
    name = "_denoise_import_smoke"
    spec = importlib.util.spec_from_file_location(name, target)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load {target}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    print(f"OK: imported {target.name} (module {mod.__name__})")


if __name__ == "__main__":
    main()

# ProX Development Guide

## Cursor Cloud specific instructions

### Overview

This is **ProX** ("Programming Every Example"), an ML research codebase for pre-training data refinement using language models. It contains three main sub-projects:

| Component | Location | Purpose |
|-----------|----------|---------|
| **ProX core** | `/workspace` (root) | Data refining pipeline (doc/chunk level), training, eval scripts |
| **VeOmni** | `/workspace/finetuning/veomni` | ByteDance distributed training framework (vendored) |
| **LightEval** | `/workspace/lighteval` | HuggingFace evaluation framework (vendored) |

### Python Environments

Each component has its own virtual environment. The Cloud Agent VM has **no GPU**, so all environments use CPU-only PyTorch.

| Component | Venv | Python | Activation |
|-----------|------|--------|------------|
| ProX core | `.venv-prox` | 3.11 | `source /workspace/.venv-prox/bin/activate` |
| VeOmni | `finetuning/veomni/.venv` | 3.11 | `source /workspace/finetuning/veomni/.venv/bin/activate` |
| LightEval | `.venv-lighteval` | 3.11 | `source /workspace/.venv-lighteval/bin/activate` |

### Lint / Test / Build Commands

**ProX core**: No formal lint/test framework. Verify imports with `python -c "import torch, transformers, datatrove, nltk"`.

**VeOmni** (see `finetuning/veomni/AGENTS.md` for full details):
```bash
cd /workspace/finetuning/veomni
source .venv/bin/activate
make quality    # ruff check + format check
make style      # ruff fix + format
pytest tests/   # run tests (many require GPU/distributed — skip those on CPU)
```

**LightEval**:
```bash
cd /workspace/lighteval
source /workspace/.venv-lighteval/bin/activate
ruff check src/lighteval/
pytest tests/unit/ --ignore=tests/unit/metrics/test_automated_metrics_pytest.py
```

### Important Caveats

- **No GPU available**: Training, inference (vLLM), and GPU-dependent tests will not run. Only CPU-based utilities, data processing logic, lint, and non-GPU tests work.
- **VeOmni torch install**: After `uv sync --dev`, torch is NOT automatically installed (it's only in the `gpu`/`npu` extras). CPU-only torch must be installed separately via `pip install torch --index-url https://download.pytorch.org/whl/cpu` using the venv's pip (`/workspace/finetuning/veomni/.venv/bin/pip`).
- **VeOmni pre-existing lint issues**: `make quality` reports a minor import sort issue and 4 format issues in committed code.
- **VeOmni tests**: `tests/utils/test_rank0_load_and_broadcast_weights.py` and `tests/utils/test_count_flops.py` fail at collection time on CPU (distributed backend / model type issues). The `tests/ops/` suite requires `triton` (GPU-only).
- **LightEval test_automated_metrics_pytest.py**: Fails at collection due to a `pytest.skip` usage issue (pre-existing).
- **uv version**: VeOmni pins `uv==0.9.8` in `pyproject.toml`. Use `~/.local/bin/uv`.

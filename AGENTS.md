# Project Guidelines

## Architecture
- Notebook-first ML workflow lives at repo root: `01_Model_Development.ipynb` for training/validation and `02_Model_Deployment.ipynb` for inference/submission.
- Reusable helpers are in `Data/preprocessing.py` (`DataProcessor`) and `Data/gene_query.py` (`GeneQuery`).
- Persisted artifacts are stored in `artifacts/` and as `pipeline_artifacts*.joblib` at root.
- `hackathon/` is a separate minimal Python package managed by `uv` (`hackathon/pyproject.toml`, `hackathon/main.py`).

## Build And Run
- Use Python 3.12+ (required by `hackathon/pyproject.toml`).
- Root notebook workflow:
```bash
code .
jupyter notebook 01_Model_Development.ipynb
jupyter notebook 02_Model_Deployment.ipynb
```
- Helper scripts (repo root):
```bash
python Data/preprocessing.py
python Data/gene_query.py
```
- `hackathon/` package workflow:
```bash
cd hackathon
uv sync
uv run python main.py
```
- There is no formal pytest/unittest suite. Validate changes with notebook reruns and targeted smoke checks.

## Code Style
- Follow typed Python signatures for public functions (`Data/preprocessing.py` is the reference style).
- Keep naming consistent: `PascalCase` classes, `snake_case` functions/variables, `UPPER_SNAKE_CASE` constants.
- Use Google/NumPy-style docstrings for non-trivial functions and include `Raises` where relevant.
- Do not use bare `except:`; catch specific exceptions and preserve context in error messages.
- Prefer logging for reusable module behavior; avoid noisy `print` in library-like code.

## Project Conventions
- Prevent leakage: fit transformers on train only, then transform validation/test data.
- Keep modality alignment explicit (transcriptomic/genomic/cell-deconvolution/ssGSEA/clinical must stay index-aligned by sample).
- Use fixed seeds for reproducibility (`random_state=42`) in splits, CV, and models unless intentionally changed.
- Prefer sklearn pipelines for preprocessing + selection + model steps.
- Use relative paths rooted at the workspace; do not introduce machine-specific absolute paths.

## Pitfalls
- `Data/preprocessing.py` and `Data/gene_query.py` import `SynOmics.*`; this package is not included in this repo, so script execution may fail without that dependency.
- `miceforest` is optional in preprocessing code (guarded import pattern); keep optional dependency guards when extending imputation logic.
- `hackathon/README.md` is currently empty, so do not treat it as a source of truth.

## Key References
- `README.md` for challenge context and modality definitions.
- `selectors.md` for feature-selection notes.
- `Data/test/`, `Data/train/`, `Data/test_nivo/`, `Data/train_nivo/` for split-specific datasets.

*Last updated: 2026-03-13*

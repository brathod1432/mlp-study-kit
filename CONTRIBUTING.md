# Contributing to mlp-study-kit

This is a personal study repository. Contributions are welcome from collaborators
who have been explicitly invited.

---

## Ground Rules

1. **Only `bgrathod00@gmail.com`** may appear as commit author or co-author on commits
   that are pushed to `main`.
2. All commits must pass the CI checks (tests + lint + security scan) before merging.
3. No `*.docx`, `*.pdf`, `*.log`, model weights (`*.h5`, `*.pt`), or data files
   should be committed. See `.gitignore`.

---

## Development Setup

```bash
# 1. Clone
git clone https://github.com/brathod1432/mlp-study-kit.git
cd mlp-study-kit

# 2. Create virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\Activate.ps1    # Windows PowerShell

# 3. Install dependencies for your platform
# Windows:
pip install -r requirements_windows.txt   # runtime + TF
# Linux:
pip install -r requirements_linux.txt     # runtime + TF

# Dev tools (both platforms):
pip install -r requirements_dev.txt

# 4. Install nn_core as editable package (no PYTHONPATH needed)
pip install -e .
```

---

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src/nn_core --cov-report=term-missing

# Single file
pytest tests/test_network.py -v
```

---

## Linting

```bash
ruff check src/nn_core/
```

---

## Security Scan

```bash
pip-audit -r requirements.txt
```

---

## Commit Message Convention

```
[MLP] <short imperative description>

Optional longer explanation.
```

Examples:
- `[MLP] Add ELU activation derivative test`
- `[MLP] Fix early stopping epsilon comparison`
- `[MLP] Update nn_core/network.py type hints`

---

## Reporting Issues

Open a GitHub Issue at https://github.com/brathod1432/mlp-study-kit/issues.
For security issues, see `SECURITY.md`.

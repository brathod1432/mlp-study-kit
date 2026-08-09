# Contributing to mlp-study-kit

mlp-study-kit is released under the **MIT License** — you are free to use,
copy, modify, and distribute it for any purpose.

Bug reports, feature ideas, and pull requests from anyone are genuinely welcome.
The only rule is that **only [@brathod1432](https://github.com/brathod1432)**
reviews and merges changes into `main`.

---

## Ways to Contribute

| Contribution | Who can do it |
|---|---|
| Open a bug report or feature request | Anyone |
| Ask a question via an Issue | Anyone |
| Submit a Pull Request | Anyone — reviewed by @brathod1432 |
| Merge a PR into `main` | @brathod1432 only |
| Push directly to `main` | @brathod1432 only |

---

## Reporting Bugs or Suggesting Improvements

Open a GitHub Issue at:
**https://github.com/brathod1432/mlp-study-kit/issues**

Please include:
- Python version and OS (Windows / Linux)
- Which file or exercise you were running
- The full error message or unexpected output
- Steps to reproduce

For security vulnerabilities, **do not open a public issue** — see [SECURITY.md](SECURITY.md).

---

## Submitting a Pull Request

1. **Fork** the repository on GitHub
2. Create a feature branch: `git checkout -b fix/describe-your-change`
3. Set up your environment:

```bash
# Clone your fork
git clone https://github.com/<your-username>/mlp-study-kit.git
cd mlp-study-kit

# Create a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate           # Linux / macOS
.venv\Scripts\Activate.ps1         # Windows PowerShell

# Install deps for your platform
pip install -r requirements_windows.txt   # Windows
pip install -r requirements_linux.txt     # Linux

# Dev tools (both platforms)
pip install -r requirements_dev.txt

# Install nn_core as editable package
pip install -e .

# Install pre-commit hooks
pre-commit install
```

4. Make your changes
5. Run all checks before pushing:

```bash
pytest                              # all 54 tests must pass
ruff check src/nn_core/             # no lint errors
bandit -r src/ -c pyproject.toml   # no security issues
```

6. Commit with the project convention:

```
[MLP] <short imperative description>

Optional longer explanation.
```

Examples:
- `[MLP] Add softmax activation to nn_core`
- `[MLP] Fix typo in ex09 backprop comment`
- `[MLP] Add test for save_weights shape mismatch`

7. Push your branch and open a Pull Request against `main`

---

## What Makes a Good PR

- **One focused change** per PR (easier to review)
- **Tests pass** — `pytest` must be green
- **No new lint errors** — `ruff check src/nn_core/`
- **No `.docx`, `.pdf`, `.log`, or model weight files** committed
- If you add a new public method to `nn_core`, add a test for it

---

## Code Style

- `nn_core/` follows `ruff` rules (see `[tool.ruff.lint]` in `pyproject.toml`)
- Type hints required for all public methods in `src/nn_core/`
- Docstrings required for all public methods in `src/nn_core/`
- Exercises (`exercises/`) are intentionally standalone and lower-polish — minor style issues are acceptable there

---

## License

By submitting a contribution you agree that your changes will be released
under the same **MIT License** that covers the rest of this project.
See [LICENSE](LICENSE) for the full text.

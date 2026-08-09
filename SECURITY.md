# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| `main` branch | Yes |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, **do not open a public issue**.

Please email **bgrathod00@gmail.com** with:
- A clear description of the issue
- Steps to reproduce
- Potential impact
- (Optional) A suggested fix

You can expect an acknowledgement within 48 hours.

## Dependency Scanning

This project uses [pip-audit](https://github.com/pypa/pip-audit) for automated CVE scanning.
The CI pipeline runs `pip-audit -r requirements.txt` on every push to `main`.

To run locally:
```bash
pip install pip-audit
pip-audit -r requirements.txt --skip-editable
```

## Known Constraints

- `numpy < 2.0` is pinned intentionally for API compatibility with TF 2.x
- `tensorflow >= 2.14, < 3.0` is pinned to avoid the Keras 3 API break

## Security Best Practices for Contributors

1. **No secrets in code** -- never commit API keys, passwords, tokens, or credentials
2. **No `eval()` / `exec()`** -- not used in `nn_core`; do not introduce them
3. **No arbitrary file writes** based on user input in library code
4. Use `raise ValueError` (not `sys.exit()`) for invalid inputs in library functions
5. Keep `requirements.txt` pinned with upper bounds -- avoid floating `>=` without `<`

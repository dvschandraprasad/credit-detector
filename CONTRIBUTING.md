# Contributing

Thanks for your interest in Credit Detector.

## Important: the algorithm is frozen

`credit_detector.py` is the code that was submitted and graded for a university
course (Group 12). Please **do not change its logic, thresholds or behaviour** in
the main branch. Known problems are listed under
[Known limitations](README.md#known-limitations) on purpose.

Welcome changes:

- Fixes and improvements to the documentation (`README.md`, `docs/`).
- New or better tests in `tests/`.
- Example videos or screenshots (screenshots go in `docs/images/`; videos stay out of git).
- Tooling such as CI.

If you want to improve the algorithm itself, please do it in a **fork or a
separate branch**, and say clearly in the README that it differs from the graded version.

## How to work on it

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest
```

All tests must pass before you open a pull request. GitHub Actions runs them on
Python 3.9 to 3.12.

## Pull requests

1. Make a branch from `main`.
2. Keep each pull request small and about one thing.
3. Describe what you changed and why.
4. Add a line to `CHANGELOG.md` if users will notice the change.

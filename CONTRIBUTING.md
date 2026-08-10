# Contributing

Thank you for helping improve Calcium Imaging Dashboard.

## Before opening an issue

- Search existing issues to avoid duplicates.
- Include your operating system, Python version, and the application command
  you ran.
- Provide a minimal example or anonymized sample when possible. Do not attach
  confidential or personally identifying research data.
- Report security concerns privately as described in `SECURITY.md`.

## Development setup

Use Python 3.11 or 3.12 and install the project with its test dependencies:

```bash
python -m venv .venv
```

Activate the environment, then run:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest
```

On Windows, activate the environment with `.venv\Scripts\activate`. On Linux
and macOS, use `source .venv/bin/activate`.

## Pull requests

- Keep each pull request focused on one change.
- Add or update tests for behavior changes.
- Update the README or in-app documentation when user-facing behavior changes.
- Add noteworthy changes under `Unreleased` in `CHANGELOG.md`.
- Confirm that `python -m pytest` passes before requesting review.

By submitting a contribution, you agree that it is licensed under the
repository's BSD 3-Clause License.

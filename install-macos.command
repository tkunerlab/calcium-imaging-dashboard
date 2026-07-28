#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

PYTHON_BIN=""
for candidate in python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)'; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python 3.11 or 3.12 is required."
  echo "Install Python from https://www.python.org/downloads/ and run this file again."
  read -r -p "Press Return to close."
  exit 1
fi

"$PYTHON_BIN" -m venv --clear .venv
.venv/bin/python -m pip install .

printf '\nInstallation complete.\n'
printf 'Run .venv/bin/cell-registration-dashboard to launch the dashboard.\n'
printf 'Run .venv/bin/db-builder to launch the database builder.\n'
read -r -p "Press Return to close."

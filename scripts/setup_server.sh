#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${CONDA_DEFAULT_ENV:-}" != "ex" ]]; then
  printf '%s\n' 'Activate the requested conda environment ex before running this script.' >&2
  exit 2
fi
mkdir -p artifacts/setup
if [[ ! -f artifacts/setup/ex-before.txt ]]; then
  python -m pip freeze > artifacts/setup/ex-before.txt
fi
python -m pip install --no-cache-dir -c artifacts/setup/ex-before.txt -r requirements-server.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python -m pip freeze > artifacts/setup/ex-after.txt
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m ir2_benchmark.cli doctor --require-gpu --output artifacts/setup/doctor.json
python -m pytest tests/benchmark -q
printf '%s\n' 'SERVER_SETUP_PASSED'

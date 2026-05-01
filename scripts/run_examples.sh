#!/usr/bin/env bash
set -euo pipefail

for example in examples/[0-9]*.py; do
  echo "--- ${example} ---"
  PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}" python "${example}"
done

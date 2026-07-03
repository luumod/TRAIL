#!/usr/bin/env bash
set -euo pipefail
python src/main_train.py --dataset mimic-iv --device "${1:-cuda:0}" --epochs "${EPOCHS:-30}" --commit "${COMMIT:-mimiciv_run}"

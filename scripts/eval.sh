#!/usr/bin/env bash
set -euo pipefail

config="${1:?Usage: $0 <config.yaml> [evaluation options]}"
shift
config="$(realpath "${config}")"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/third_party/transformers/src${PYTHONPATH:+:${PYTHONPATH}}"
export LMMS_EVAL_LAUNCHER="${LMMS_EVAL_LAUNCHER:-accelerate}"
export VSIBENCH_REFINED_EVAL="${VSIBENCH_REFINED_EVAL:-true}"

# Use an installed lmms-eval package, or an optional local source checkout.
lmms_eval_root="${VIPS_LMMS_EVAL_ROOT:-${repo_root}/third_party/lmms-eval}"
if [[ -d "${lmms_eval_root}/lmms_eval" ]]; then
  export PYTHONPATH="${lmms_eval_root}${PYTHONPATH:+:${PYTHONPATH}}"
fi

exec accelerate launch --num_processes="${VIPS_NUM_GPUS:-1}" \
  -m vips.eval --config "${config}" "$@"

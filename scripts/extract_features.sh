#!/usr/bin/env bash
set -euo pipefail

config="${1:?Usage: $0 <config.yaml> [train|vsibench] [extraction options]}"
shift
config="$(realpath "${config}")"
split="${1:-vsibench}"
if [[ $# -gt 0 ]]; then
  shift
fi

[[ "${split}" == train || "${split}" == vsibench ]] || {
  echo "Usage: $0 <config.yaml> [train|vsibench] [extraction options]" >&2
  exit 2
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/third_party/transformers/src${PYTHONPATH:+:${PYTHONPATH}}"
exec torchrun --standalone --nproc_per_node="${VIPS_NUM_GPUS:-1}" \
  -m vips.feature_extraction "${config}" "${split}" "$@"

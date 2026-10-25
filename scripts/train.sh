#!/usr/bin/env bash
set -euo pipefail

config="${1:?Usage: $0 <config.yaml> [training options]}"
shift
config="$(realpath "${config}")"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${repo_root}"
export PYTHONPATH="${repo_root}/third_party/transformers/src${PYTHONPATH:+:${PYTHONPATH}}"
exec torchrun --nproc_per_node="${VIPS_NUM_GPUS:-8}" \
  -m vips.train --config "${config}" "$@"

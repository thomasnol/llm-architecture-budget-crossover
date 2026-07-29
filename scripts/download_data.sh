#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="${repo_dir}/data/raw"
commit="9aa8782f850a41de2e7d21edf4def91ce99c0d08"
base="https://huggingface.co/datasets/snorkelai/Multi-Turn-Insurance-Underwriting/resolve/${commit}"

mkdir -p "${data_dir}"
curl -fsSL "${base}/data/train-00000-of-00001.parquet" -o "${data_dir}/train.parquet"
curl -fsSL "${base}/README.md" -o "${data_dir}/DATASET_CARD.md"

expected="55833ec064222f8a98a80af8e9726ad98f8540f8173be97343e50bac3fb37c83"
actual="$(sha256sum "${data_dir}/train.parquet" | awk '{print $1}')"
if [[ "${actual}" != "${expected}" ]]; then
  echo "Checksum mismatch: expected ${expected}, got ${actual}" >&2
  exit 1
fi
echo "Downloaded dataset commit ${commit} (${actual})."

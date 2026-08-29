#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "$script_dir/.." && pwd)"
raw_dir="$experiment_dir/data/raw"
archive="$raw_dir/PMC11285529_SupplementaryFiles.zip"
workbook="$raw_dir/12870_2024_5394_MOESM5_ESM.xlsx"
url="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11285529/supplementaryFiles"

mkdir -p "$raw_dir"
if [[ ! -s "$archive" ]]; then
  curl -L --fail --show-error --silent "$url" -o "$archive"
fi
if [[ ! -s "$workbook" ]]; then
  unzip -j -o "$archive" '12870_2024_5394_MOESM5_ESM.xlsx' -d "$raw_dir" >/dev/null
fi

printf '%s\n' "$workbook"

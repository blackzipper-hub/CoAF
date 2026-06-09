#!/usr/bin/env bash
# Rewrite manifest paths from old prefix to Casual_CoAF/coaf_dataset (optional fallback).
set -euo pipefail

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fix_dataset_paths.sh"

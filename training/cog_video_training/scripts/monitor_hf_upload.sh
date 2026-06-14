#!/usr/bin/env bash
# Monitor HF dataset upload progress. Logs to logs/hf/coaf_dataset_24_25_monitor.log
set -euo pipefail

CASUAL_ROOT="${CASUAL_ROOT:-/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training}"
LOG="${CASUAL_ROOT}/logs/hf/coaf_dataset_24_25_monitor.log"
REPO_ID="${HF_REPO_ID:-BlackZipper/coaf_dataset_24_25}"
UPLOAD_LOG="${CASUAL_ROOT}/logs/hf/coaf_dataset_24_25_tmux_upload.log"
INTERVAL="${MONITOR_INTERVAL_SEC:-120}"
TMUX_SESSION="${UPLOAD_TMUX_SESSION:-hfup_coaf25}"

mkdir -p "${CASUAL_ROOT}/logs/hf"

while true; do
  ts="$(date -Is)"
  {
    echo "=== ${ts} ==="

    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
      echo "tmux: ${TMUX_SESSION} running"
    else
      echo "tmux: ${TMUX_SESSION} NOT running"
    fi

    if pgrep -f 'upload_coaf_dataset_to_hf.py' >/dev/null 2>&1; then
      ps -o pid=,pcpu=,pmem=,etime=,stat= -C python 2>/dev/null \
        | grep -F 'upload_coaf_dataset' || ps aux | grep '[u]pload_coaf_dataset_to_hf.py' || true
      echo "process: running"
    else
      echo "process: NOT running"
    fi

    if [[ -f "${UPLOAD_LOG}" ]]; then
      echo "upload_log_bytes: $(wc -c < "${UPLOAD_LOG}")"
      tail -3 "${UPLOAD_LOG}" | sed 's/^/  /'
    fi

    if [[ -n "${HF_TOKEN:-}" ]]; then
      hub_json="$(HF_TOKEN="${HF_TOKEN}" hf datasets info "${REPO_ID}" --format json 2>/dev/null || true)"
      if [[ -n "${hub_json}" ]]; then
        python3 -c "
import json, sys
d = json.loads(sys.argv[1])
sibs = d.get('siblings', [])
print('hub_files:', len(sibs))
print('used_storage:', d.get('used_storage', 0))
" "${hub_json}"
      fi
    else
      echo "hub: skipped (HF_TOKEN not set)"
    fi
    echo
  } >> "${LOG}" 2>&1

  sleep "${INTERVAL}"
done

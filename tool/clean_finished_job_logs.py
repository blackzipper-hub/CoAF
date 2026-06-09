#!/usr/bin/env python3
"""
Read and clean SLURM job logs under Casual_CoAF training logs.

Only processes jobs that are no longer running (via squeue/sacct).
Typical cleanup: strip ANSI codes, collapse tqdm progress spam (Steps / Loading weights).

Usage:
  python tool/clean_finished_job_logs.py --dry-run
  python tool/clean_finished_job_logs.py
  python tool/clean_finished_job_logs.py --log-root /path/to/logs --in-place
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# jobname-12345.out | jobname-12345_3.err
LOG_NAME_RE = re.compile(r"^.+-(?P<job_id>\d+)(?:_(?P<array_id>\d+))?\.(?:out|err)$")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
STEPS_RE = re.compile(r"^Steps:\s+\d+%\|.*\|\s*(\d+)/")
LOADING_WEIGHTS_RE = re.compile(r"^Loading weights:\s+\d+%\|")

FINISHED_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }
)
ACTIVE_STATES = frozenset(
    {
        "CONFIGURING",
        "PENDING",
        "RUNNING",
        "REQUEUED",
        "RESIZING",
        "REVOKED",
        "SUSPENDED",
    }
)

DEFAULT_LOG_ROOT = (
    Path(__file__).resolve().parents[1]
    / "training"
    / "cog_video_training"
    / "logs"
)


@dataclass(frozen=True)
class LogFile:
    path: Path
    slurm_job_id: str  # e.g. "432840" or "434304_0"


def parse_slurm_job_id(path: Path) -> str | None:
    match = LOG_NAME_RE.match(path.name)
    if not match:
        return None
    job_id = match.group("job_id")
    array_id = match.group("array_id")
    if array_id is not None:
        return f"{job_id}_{array_id}"
    return job_id


def run_cmd(argv: list[str]) -> str:
    try:
        proc = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"command failed: {' '.join(argv)}") from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(
            f"command exited {proc.returncode}: {' '.join(argv)}\n{stderr}"
        )
    return proc.stdout


def get_running_job_ids(user: str) -> set[str]:
    out = run_cmd(
        [
            "squeue",
            "-u",
            user,
            "-h",
            "-o",
            "%i",
        ]
    )
    running: set[str] = set()
    for line in out.splitlines():
        job_id = line.strip()
        if not job_id:
            continue
        running.add(job_id)
        if "." in job_id:
            running.add(job_id.split(".", 1)[0])
    return running


def normalize_sacct_state(state: str) -> str:
    # sacct may print CANCELLED+ or COMPLETED+
    return state.rstrip("+").strip()


def get_sacct_states(job_ids: set[str], user: str) -> dict[str, str]:
    if not job_ids:
        return {}
    # sacct -j accepts comma-separated list; chunk to avoid argv limits
    states: dict[str, str] = {}
    sorted_ids = sorted(job_ids, key=lambda x: (int(x.split("_")[0]), x))
    chunk_size = 200
    for i in range(0, len(sorted_ids), chunk_size):
        chunk = sorted_ids[i : i + chunk_size]
        out = run_cmd(
            [
                "sacct",
                "-u",
                user,
                "-n",
                "-P",
                "-j",
                ",".join(chunk),
                "--format=JobID,State",
            ]
        )
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            job_id, state = parts[0].strip(), normalize_sacct_state(parts[1])
            if job_id.endswith(".batch") or job_id.endswith(".extern"):
                continue
            states[job_id] = state
    return states


def is_job_finished(
    slurm_job_id: str,
    running: set[str],
    sacct_states: dict[str, str],
) -> bool:
    base_id = slurm_job_id.split("_")[0]
    if slurm_job_id in running or base_id in running:
        return False

    state = sacct_states.get(slurm_job_id)
    if state is None and "_" in slurm_job_id:
        state = sacct_states.get(base_id)
    if state is None:
        # Unknown to sacct (expired from accounting): treat as finished if not running
        return True
    if state in ACTIVE_STATES:
        return False
    if state in FINISHED_STATES:
        return True
    # Conservative default for odd states
    return state not in ACTIVE_STATES


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def clean_log_text(raw: str) -> tuple[str, dict[str, int]]:
    stats = {
        "lines_in": 0,
        "lines_out": 0,
        "steps_collapsed": 0,
        "loading_collapsed": 0,
        "ansi_stripped": 0,
    }
    out_lines: list[str] = []
    last_step: str | None = None
    last_loading_kept = False

    for raw_line in raw.splitlines():
        stats["lines_in"] += 1
        line = strip_ansi(raw_line.rstrip("\r\n"))
        if line != raw_line.rstrip("\r\n"):
            stats["ansi_stripped"] += 1

        if STEPS_RE.match(line):
            step = STEPS_RE.match(line).group(1)
            if step == last_step:
                stats["steps_collapsed"] += 1
                if out_lines and STEPS_RE.match(out_lines[-1]):
                    out_lines[-1] = line
                continue
            last_step = step
            out_lines.append(line)
            continue

        if LOADING_WEIGHTS_RE.match(line):
            is_done = "100%|" in line
            if last_loading_kept and not is_done:
                stats["loading_collapsed"] += 1
                if out_lines and LOADING_WEIGHTS_RE.match(out_lines[-1]):
                    out_lines[-1] = line
                continue
            last_loading_kept = not is_done
            out_lines.append(line)
            continue

        out_lines.append(line)

    stats["lines_out"] = len(out_lines)
    cleaned = "\n".join(out_lines)
    if raw.endswith("\n"):
        cleaned += "\n"
    return cleaned, stats


def discover_logs(log_root: Path) -> list[LogFile]:
    logs: list[LogFile] = []
    for path in sorted(log_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".out", ".err"}:
            continue
        if path.name.endswith(".bak") or path.name.endswith(".cleaned"):
            continue
        job_id = parse_slurm_job_id(path)
        if job_id is None:
            continue
        logs.append(LogFile(path=path, slurm_job_id=job_id))
    return logs


def format_bytes(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.1f}{unit}" if unit != "B" else f"{num}B"
        num /= 1024
    return f"{num:.1f}TB"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean SLURM logs for finished jobs under Casual_CoAF logs/"
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help=f"Root log directory (default: {DEFAULT_LOG_ROOT})",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("USER") or os.environ.get("LOGNAME"),
        help="SLURM user for squeue/sacct (default: $USER)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be cleaned, do not write files",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite original log files (default: write *.cleaned alongside)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="With --in-place, skip creating .bak backups",
    )
    parser.add_argument(
        "--include-running",
        action="store_true",
        help="Also clean logs for jobs still running (not recommended)",
    )
    args = parser.parse_args()

    if not args.user:
        print("Cannot determine SLURM user; pass --user", file=sys.stderr)
        return 1

    log_root = args.log_root.resolve()
    if not log_root.is_dir():
        print(f"Log root does not exist: {log_root}", file=sys.stderr)
        return 1

    log_files = discover_logs(log_root)
    if not log_files:
        print(f"No .out/.err logs found under {log_root}")
        return 0

    job_ids = {lf.slurm_job_id for lf in log_files}
    try:
        running = get_running_job_ids(args.user)
        sacct_states = get_sacct_states(job_ids, args.user)
    except RuntimeError as exc:
        print(f"SLURM query failed: {exc}", file=sys.stderr)
        print(
            "Ensure squeue/sacct are available on the cluster login/submit node.",
            file=sys.stderr,
        )
        return 1

    total_in_bytes = 0
    total_out_bytes = 0
    cleaned_count = 0
    skipped_running = 0
    skipped_unchanged = 0

    for lf in log_files:
        if not args.include_running and not is_job_finished(
            lf.slurm_job_id, running, sacct_states
        ):
            skipped_running += 1
            continue

        raw = lf.path.read_text(encoding="utf-8", errors="replace")
        cleaned, stats = clean_log_text(raw)
        if cleaned == raw:
            skipped_unchanged += 1
            continue

        in_size = len(raw.encode("utf-8"))
        out_size = len(cleaned.encode("utf-8"))
        total_in_bytes += in_size
        total_out_bytes += out_size
        cleaned_count += 1

        state = sacct_states.get(lf.slurm_job_id, "?")
        rel = lf.path.relative_to(log_root)
        print(
            f"{'[dry-run] ' if args.dry_run else ''}{rel} "
            f"job={lf.slurm_job_id} state={state} "
            f"{format_bytes(in_size)} -> {format_bytes(out_size)} "
            f"(steps -{stats['steps_collapsed']}, "
            f"loading -{stats['loading_collapsed']})"
        )

        if args.dry_run:
            continue

        if args.in_place:
            if not args.no_backup:
                backup = lf.path.with_suffix(lf.path.suffix + ".bak")
                if not backup.exists():
                    shutil.copy2(lf.path, backup)
            lf.path.write_text(cleaned, encoding="utf-8")
        else:
            out_path = lf.path.with_suffix(lf.path.suffix + ".cleaned")
            out_path.write_text(cleaned, encoding="utf-8")

    print()
    print(f"Scanned {len(log_files)} log files under {log_root}")
    print(f"Skipped (still running): {skipped_running}")
    print(f"Skipped (already clean): {skipped_unchanged}")
    print(f"{'Would clean' if args.dry_run else 'Cleaned'}: {cleaned_count}")
    if cleaned_count:
        saved = total_in_bytes - total_out_bytes
        print(
            f"Size: {format_bytes(total_in_bytes)} -> {format_bytes(total_out_bytes)} "
            f"(saved {format_bytes(saved)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

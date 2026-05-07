#!/usr/bin/env python3
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Summary:
    path: Path
    checkpoint: str | None
    model_family: str | None
    task_suite: str | None
    num_trials_per_task: int | None
    seed: int | None
    total_success_rate: float | None


def _parse_header_field(text: str, prefix: str) -> str | None:
    m = re.search(rf"^{re.escape(prefix)}\s*(.*)\s*$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None


def _parse_int_field(text: str, prefix: str) -> int | None:
    val = _parse_header_field(text, prefix)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def parse_log(path: Path) -> Summary:
    text = path.read_text(errors="replace")
    checkpoint = _parse_header_field(text, "# Checkpoint (pretrained_checkpoint):")
    model_family = _parse_header_field(text, "# Model family:")
    task_suite_raw = _parse_header_field(text, "# Task suite:")
    task_suite = task_suite_raw.split(" ", 1)[0] if task_suite_raw else None
    num_trials = _parse_int_field(text, "# Rollouts per task (num_trials_per_task):")
    seed = _parse_int_field(text, "# Misc: seed=")  # This line is "seed=7, run_id_note=..."
    if seed is None:
        m = re.search(r"# Misc:\s*seed=(\d+)\b", text)
        seed = int(m.group(1)) if m else None

    # Take the final "Current total success rate" printed once per task.
    rates = re.findall(r"^Current total success rate:\s*([0-9]*\.?[0-9]+)\s*$", text, flags=re.MULTILINE)
    total_success_rate = float(rates[-1]) if rates else None

    return Summary(
        path=path,
        checkpoint=checkpoint,
        model_family=model_family,
        task_suite=task_suite,
        num_trials_per_task=num_trials,
        seed=seed,
        total_success_rate=total_success_rate,
    )


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: parse_libero_eval_log.py LOG1 [LOG2 ...]")
        return 2

    summaries = [parse_log(Path(p)) for p in argv[1:]]
    for s in summaries:
        rate = "N/A" if s.total_success_rate is None else f"{s.total_success_rate*100:.2f}%"
        print(f"{s.path}")
        print(f"  checkpoint: {s.checkpoint or 'N/A'}")
        print(f"  model_family: {s.model_family or 'N/A'}")
        print(f"  task_suite: {s.task_suite or 'N/A'}")
        print(f"  num_trials_per_task: {s.num_trials_per_task if s.num_trials_per_task is not None else 'N/A'}")
        print(f"  seed: {s.seed if s.seed is not None else 'N/A'}")
        print(f"  total_success_rate: {rate}")
        print()

    # If exactly 2 logs, print a delta.
    if len(summaries) == 2 and all(s.total_success_rate is not None for s in summaries):
        a, b = summaries
        delta = (b.total_success_rate - a.total_success_rate) * 100.0
        print(f"Delta (2nd - 1st): {delta:+.2f} percentage points")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

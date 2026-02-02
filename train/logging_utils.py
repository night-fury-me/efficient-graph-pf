from __future__ import annotations

import os
import sys
from dataclasses import dataclass


class TeeStdout:
    """Write stdout to terminal and a file (like `tee`)."""

    def __init__(self, filename: str):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


@dataclass(frozen=True)
class RunPaths:
    results_dir: str = "./results"
    ckpt_dir: str = "./results/ckpt"
    plots_dir: str = "./results/plots"


def ensure_run_dirs(paths: RunPaths = RunPaths()) -> None:
    os.makedirs(paths.ckpt_dir, exist_ok=True)
    os.makedirs(paths.plots_dir, exist_ok=True)


def redirect_stdout_to_log(runname: str, results_dir: str = "./results") -> str:
    log_filename = os.path.join(results_dir, f"{runname}_training_log.txt")
    sys.stdout = TeeStdout(log_filename)  # type: ignore[assignment]
    return log_filename

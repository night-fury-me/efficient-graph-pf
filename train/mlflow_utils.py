from __future__ import annotations

import logging
import os
import subprocess
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger("simplegnn")


def _quiet_mlflow_deps() -> None:
    """Reduce verbosity of MLflow's dependencies (Alembic/SQLAlchemy).

    MLflow can emit a burst of INFO logs on startup (esp. when using
    sqlite:///...), which tends to drown out training logs.
    """

    if os.getenv("QUIET_THIRD_PARTY", "1") in {"0", "false", "False"}:
        return

    for noisy in (
        "alembic",
        "alembic.runtime.migration",
        "alembic.runtime.plugins",
        "sqlalchemy",
        "mlflow",
        "mlflow.store",
        "mlflow.store.db",
        "mlflow.store.db.utils",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _try_import_mlflow():
    try:
        import mlflow  # type: ignore

        return mlflow
    except Exception as e:
        log.warning("MLflow is not available (%s). Proceeding without MLflow.", e)
        return None


def _git_sha_short(repo_root: Path) -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        sha = (res.stdout or "").strip()
        return sha or None
    except Exception:
        return None


def snapshot_code(*, repo_root: Path, out_zip: Path, include_globs: Iterable[str]) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()

    paths: list[Path] = []
    for pat in include_globs:
        paths.extend(repo_root.glob(pat))

    # Deduplicate + keep files only
    seen: set[Path] = set()
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            continue
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        files.append(p)

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(files, key=lambda x: str(x)):
            try:
                arc = str(p.relative_to(repo_root))
            except Exception:
                arc = p.name
            zf.write(p, arcname=arc)


@contextmanager
def mlflow_run(
    *,
    enabled: bool,
    tracking_uri: Optional[str],
    experiment: str,
    run_name: str,
    tags: Optional[dict[str, str]] = None,
):
    mlflow = _try_import_mlflow()
    if not enabled or mlflow is None:
        yield None
        return

    # MLflow (or its deps) may configure logging; enforce our desired noise level.
    _quiet_mlflow_deps()

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name):
        if tags:
            for k, v in tags.items():
                try:
                    mlflow.set_tag(k, v)
                except Exception:
                    pass
        yield mlflow


def log_run_artifacts(
    *,
    mlflow,
    run_dir: Path,
    artifact_path: str = "run",
) -> None:
    try:
        mlflow.log_artifacts(str(run_dir), artifact_path=artifact_path)
    except Exception as e:
        log.warning("Failed to log artifacts to MLflow: %s", e)


def log_params_safe(mlflow, params: dict) -> None:
    for k, v in params.items():
        try:
            mlflow.log_param(k, v)
        except Exception:
            # MLflow can reject long strings or unsupported types.
            try:
                mlflow.log_param(k, str(v))
            except Exception:
                pass


def log_metrics_history(mlflow, *, history: dict[str, list[float]]) -> None:
    # Expect 1-indexed epoch lists
    for name, series in history.items():
        for step, value in enumerate(series, start=1):
            try:
                mlflow.log_metric(name, float(value), step=step)
            except Exception:
                pass


def add_basic_tags(*, repo_root: Path) -> dict[str, str]:
    tags: dict[str, str] = {}
    sha = _git_sha_short(repo_root)
    if sha:
        tags["git_sha"] = sha
    tags["cwd"] = str(repo_root)
    tags["host"] = os.uname().nodename if hasattr(os, "uname") else "unknown"
    return tags

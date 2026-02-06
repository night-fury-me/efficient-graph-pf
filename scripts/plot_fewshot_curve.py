#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("plot_fewshot_curve")


def _try_import_mlflow():
    try:
        import mlflow  # type: ignore
        from mlflow.tracking import MlflowClient  # type: ignore

        return mlflow, MlflowClient
    except Exception as e:
        raise RuntimeError(
            "MLflow is required for this script. Install it (it is already in pyproject.toml) and ensure the environment is active."
        ) from e


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Collect few-shot adaptation results from MLflow and plot a line chart for Full-FT vs LoRA-FT. "
            "Assumes runs log a numeric metric named 'target_budget' (0..1) and your chosen test metric (e.g. test/rmse)."
        )
    )

    p.add_argument(
        "--tracking-uri",
        default="sqlite:///results/mlflow.db",
        help="MLflow tracking URI (default: sqlite:///results/mlflow.db)",
    )

    p.add_argument(
        "--experiment",
        action="append",
        required=True,
        help=(
            "Experiment name(s) to include. Provide twice for two curves, e.g. --experiment full_ft_hv --experiment lora_ft_hv_r2_a8"
        ),
    )

    p.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional labels corresponding to each --experiment (same count).",
    )

    p.add_argument(
        "--metric",
        default="test/rmse",
        help="Metric key to plot on y-axis (default: test/rmse)",
    )

    p.add_argument(
        "--out-dir",
        default="results/fewshot",
        help="Output directory for CSV + plots (default: results/fewshot)",
    )

    p.add_argument(
        "--title",
        default=None,
        help="Optional plot title.",
    )

    p.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help=(
            "Output formats to write (default: png). Example: --formats pdf svg. "
            "Supported by matplotlib backend (commonly: png pdf svg)."
        ),
    )

    p.add_argument(
        "--stem",
        default="fewshot_curve",
        help="Output filename stem (no extension). Files are written under --out-dir.",
    )

    p.add_argument(
        "--error-style",
        default="band",
        choices=["band", "bars", "none"],
        help=(
            "How to visualize variability across seeds per budget: "
            "'band' draws shaded ±1 std, 'bars' draws error bars (±1 std), 'none' draws only the mean. "
            "Default: band."
        ),
    )

    p.add_argument(
        "--write-both",
        action="store_true",
        help=(
            "Write two plot variants (band + bars) so you can compare aesthetics. "
            "Outputs are suffixed with '_band' and '_bars'."
        ),
    )

    p.add_argument(
        "--show",
        action="store_true",
        help="Show plot window (if your environment supports it).",
    )

    return p


def _get_run_seed(run: Any) -> str | None:
    # Prefer tag (this repo sets tags['seed'])
    tags = getattr(getattr(run, "data", None), "tags", None) or {}
    if isinstance(tags, dict):
        s = tags.get("seed")
        if s is not None:
            return str(s)
    # Fallback to param
    params = getattr(getattr(run, "data", None), "params", None) or {}
    if isinstance(params, dict) and "seed" in params:
        return str(params.get("seed"))
    return None


def _search_runs_all(client: Any, experiment_id: str) -> list[Any]:
    """Fetch all runs for an experiment, respecting MLflow's max_results limit.

    Some MLflow backends enforce max_results <= 50000. We page through results when
    the client supports page tokens.
    """

    runs: list[Any] = []
    page_token: str | None = None
    while True:
        try:
            page = client.search_runs(
                [experiment_id],
                filter_string="",
                max_results=50000,
                page_token=page_token,
            )
        except TypeError:
            # Older MLflow versions may not support page_token.
            page = client.search_runs([experiment_id], filter_string="", max_results=50000)
            runs.extend(list(page or []))
            break

        runs.extend(list(page or []))
        next_token = getattr(page, "token", None)
        if not next_token:
            break
        page_token = str(next_token)
    return runs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.label is not None and len(args.label) != len(args.experiment):
        raise SystemExit("If provided, --label must have the same count as --experiment")

    labels = args.label or list(args.experiment)
    metric_key = str(args.metric)
    formats = [str(x).lower().lstrip(".") for x in (args.formats or [])]
    if not formats:
        formats = ["png"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mlflow, MlflowClient = _try_import_mlflow()
    mlflow.set_tracking_uri(str(args.tracking_uri))
    client = MlflowClient()

    rows: list[dict[str, Any]] = []

    for exp_name, label in zip(args.experiment, labels):
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            raise SystemExit(f"MLflow experiment not found: {exp_name}")

        # Pull all runs, then filter in Python for robustness.
        runs = _search_runs_all(client, str(exp.experiment_id))
        log.info("Experiment '%s': %d runs", exp_name, len(runs))

        for run in runs:
            data = getattr(run, "data", None)
            if data is None:
                continue

            metrics = getattr(data, "metrics", None) or {}
            tags = getattr(data, "tags", None) or {}

            if "target_budget" not in metrics and "target_budget" not in tags:
                continue

            # budget can be metric or tag
            budget_raw = metrics.get("target_budget", tags.get("target_budget"))
            try:
                budget = float(budget_raw)
            except Exception:
                continue

            if budget < 0.0 or budget > 1.0:
                continue

            if metric_key not in metrics:
                continue

            y = float(metrics[metric_key])
            seed = _get_run_seed(run)

            rows.append(
                {
                    "label": str(label),
                    "experiment": str(exp_name),
                    "run_id": str(run.info.run_id),
                    "seed": seed,
                    "budget": float(budget),
                    "metric": str(metric_key),
                    "value": float(y),
                }
            )

    if not rows:
        raise SystemExit(
            "No rows collected. Verify that your runs logged 'target_budget' and that the chosen metric exists in MLflow (e.g. test/rmse)."
        )

    df = pd.DataFrame(rows)

    long_csv = out_dir / "fewshot_runs_long.csv"
    df.to_csv(long_csv, index=False)

    summary = (
        df.groupby(["label", "budget"], as_index=False)
        .agg(mean=("value", "mean"), std=("value", "std"), n=("value", "count"))
        .sort_values(["label", "budget"])
    )

    summary_csv = out_dir / "fewshot_summary.csv"
    summary.to_csv(summary_csv, index=False)

    # Plot
    try:
        import matplotlib.pyplot as plt

        def _plot_and_save(error_style: str, stem_suffix: str | None = None) -> list[Path]:
            fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
            written: list[Path] = []

            for label in summary["label"].unique().tolist():
                part = summary[summary["label"] == label].sort_values("budget")
                x = part["budget"]
                y = part["mean"]
                yerr = part["std"].fillna(0.0)

                if error_style == "bars":
                    ax.errorbar(
                        x,
                        y,
                        yerr=yerr,
                        fmt="-o",
                        linewidth=2.0,
                        capsize=3,
                        label=str(label),
                    )
                else:
                    ax.plot(x, y, marker="o", linewidth=2.0, label=str(label))
                    if error_style == "band" and part["std"].notna().any():
                        y1 = y - yerr
                        y2 = y + yerr
                        ax.fill_between(x, y1, y2, alpha=0.15)

            ax.set_xlabel("Target train budget (fraction)")
            ax.set_ylabel(metric_key)
            ax.grid(True, alpha=0.25)
            ax.legend()
            ax.set_title(args.title or f"Few-shot adaptation curve ({metric_key})")

            fig.tight_layout()

            stem = str(args.stem)
            if stem_suffix:
                stem = f"{stem}_{stem_suffix}"

            for fmt in formats:
                out_path = out_dir / f"{stem}.{fmt}"
                fig.savefig(out_path)
                written.append(out_path)
                log.info("Wrote %s", out_path)

            if args.show:
                plt.show()
            plt.close(fig)
            return written

        written_plots: list[Path] = []
        if bool(args.write_both):
            written_plots.extend(_plot_and_save("band", stem_suffix="band"))
            written_plots.extend(_plot_and_save("bars", stem_suffix="bars"))
        else:
            written_plots.extend(_plot_and_save(str(args.error_style), stem_suffix=None))

    except Exception as e:
        log.warning("Plotting skipped (%s). CSVs are still written.", e)

    meta = {
        "tracking_uri": str(args.tracking_uri),
        "experiments": list(args.experiment),
        "labels": list(labels),
        "metric": metric_key,
        "error_style": str(args.error_style),
        "write_both": bool(args.write_both),
        "long_csv": str(long_csv),
        "summary_csv": str(summary_csv),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info("Wrote %s", long_csv)
    log.info("Wrote %s", summary_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

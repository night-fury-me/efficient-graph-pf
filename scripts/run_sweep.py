#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

# When invoked as `python scripts/run_sweep.py`, Python puts `scripts/` at the
# front of sys.path. Prepend repo root so `import train...` resolves to the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.config_loader import deep_update, load_yaml_config


log = logging.getLogger("run_sweep")


def _parse_seeds(values: list[str]) -> list[int]:
    # Supports:
    #   --seeds 42 43 44
    #   --seeds 42,43,44
    #   --seeds 42-45
    #   --seeds 42-45,999
    seeds: list[int] = []
    for raw in values:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        for part in parts:
            m = re.fullmatch(r"(\d+)-(\d+)", part)
            if m:
                a = int(m.group(1))
                b = int(m.group(2))
                step = 1 if b >= a else -1
                seeds.extend(list(range(a, b + step, step)))
            else:
                seeds.append(int(part))
    # stable unique
    seen: set[int] = set()
    out: list[int] = []
    for s in seeds:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _parse_int_list(values: list[str]) -> list[int]:
    if not values:
        return []
    return _parse_seeds(values)


def _parse_float_list(values: list[str]) -> list[float]:
    """Parse float lists/ranges for budgets.

    Supports:
      --budgets 0 0.01 0.05
      --budgets 0,0.01,0.05
      --budgets 1% 5% 10%
    """

    out: list[float] = []
    for raw in values:
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        for part in parts:
            s = part
            if s.endswith("%"):
                s = s[:-1].strip()
                out.append(float(s) / 100.0)
            else:
                out.append(float(s))
    # stable unique
    seen: set[float] = set()
    uniq: list[float] = []
    for v in out:
        if v in seen:
            continue
        seen.add(v)
        uniq.append(v)
    return uniq


def _ensure_mapping(x: Any, *, what: str) -> dict[str, Any]:
    if x is None:
        return {}
    if not isinstance(x, Mapping):
        raise TypeError(f"Expected {what} to be a mapping/dict, got: {type(x).__name__}")
    return dict(x)


def _set_nested(d: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cur: dict[str, Any] = d
    for k in path[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[path[-1]] = value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the same scenario across multiple seeds by generating per-seed config files and invoking `python -m train`. "
            "This is the recommended way to do baselines/LoRA/etc. sweeps reproducibly."
        )
    )
    p.add_argument(
        "--base",
        "--base-config",
        dest="base_config",
        required=True,
        help="Base YAML config (e.g. configs/default.yaml)",
    )
    p.add_argument(
        "--scenario",
        "--scenario-config",
        dest="scenario_config",
        default=None,
        help=(
            "Optional scenario overlay YAML (e.g. configs/scenarios/lora.yaml). "
            "This is deep-merged on top of base."
        ),
    )
    p.add_argument(
        "--scenario-name",
        default=None,
        help="Optional short scenario name used for auto run.name and output folder naming.",
    )

    p.add_argument(
        "--mlflow-experiment",
        default=None,
        help=(
            "Optional MLflow experiment name override for the generated runs. "
            "If omitted and --scenario-name is explicitly provided, the script will set mlflow.experiment to --scenario-name."
        ),
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        required=True,
        help="Seeds list/ranges (e.g. 42 43 44 | 42,43,44 | 42-45 | 42-45,999)",
    )
    p.add_argument(
        "--name-template",
        default="{scenario}_seed{seed}_b{budget}",
        help=(
            "Template used to populate run.name when base+scenario doesn't set one. "
            "Available fields: {scenario}, {seed}, {budget}, {r}, {alpha}."
        ),
    )
    p.add_argument(
        "--force-run-name",
        action="store_true",
        help="Overwrite run.name even if base/scenario already set it.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Where to write generated per-seed configs. Default: results/sweeps/<timestamp>_<scenario>/configs"
        ),
    )
    p.add_argument(
        "--lora-rs",
        nargs="+",
        default=None,
        help=(
            "Optional LoRA rank grid. Supports list/ranges (e.g. 2 4 8 | 2,4,8 | 2-8). "
            "If set, must be paired with --lora-alphas."
        ),
    )
    p.add_argument(
        "--lora-alphas",
        nargs="+",
        default=None,
        help=(
            "Optional LoRA alpha grid. Supports list/ranges (e.g. 8 16 32 | 8,16,32 | 8-32). "
            "If set, must be paired with --lora-rs."
        ),
    )

    p.add_argument(
        "--budgets",
        nargs="+",
        default=None,
        help=(
            "Optional target train budgets (few-shot). Each value is a fraction in [0,1] or a percent like '5%'. "
            "For budgets > 0, sets split.train_subset_frac. For budget=0, forces run.mode=test (zero-shot transfer)."
        ),
    )
    p.add_argument(
        "--python",
        dest="python_exe",
        default=sys.executable,
        help="Python executable to use (default: current interpreter).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands/config paths but do not execute training.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining seeds even if one seed fails.",
    )
    p.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to `python -m train` (example: -- --EPOCHS 50 --BATCH 128)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    base_path = Path(args.base_config)
    scenario_path = Path(args.scenario_config) if args.scenario_config else None

    base = _ensure_mapping(load_yaml_config(str(base_path)), what="base config")

    scenario: dict[str, Any] = {}
    if scenario_path is not None:
        scenario = _ensure_mapping(load_yaml_config(str(scenario_path)), what="scenario config")

    seeds = _parse_seeds(list(args.seeds))
    if not seeds:
        raise SystemExit("No seeds parsed; check --seeds")

    scenario_name_cli = args.scenario_name
    scenario_name = args.scenario_name
    if not scenario_name:
        if scenario_path is not None:
            scenario_name = scenario_path.stem
        else:
            scenario_name = "scenario"

    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("results") / "sweeps" / f"{timestamp}_{scenario_name}" / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_rs = _parse_int_list(list(args.lora_rs or []))
    grid_alphas = _parse_int_list(list(args.lora_alphas or []))
    has_grid = bool(grid_rs or grid_alphas)
    if (grid_rs and not grid_alphas) or (grid_alphas and not grid_rs):
        raise SystemExit("Both --lora-rs and --lora-alphas must be provided for a grid sweep.")
    grid_items = [(None, None)]
    if has_grid:
        grid_items = [(r, a) for r in grid_rs for a in grid_alphas]

    budgets = _parse_float_list(list(args.budgets or []))
    if not budgets:
        budgets = [1.0]
    for b in budgets:
        if b < 0.0 or b > 1.0:
            raise SystemExit(f"Invalid budget {b}. Must be in [0,1].")

    merged_configs: list[tuple[Path, int, float, int | None, int | None]] = []
    for seed in seeds:
        for budget in budgets:
            for r_value, alpha_value in grid_items:
                # IMPORTANT: use deep copies so nested edits (e.g. setting run.mode=test for budget=0)
                # do not mutate the shared base dict and leak into subsequent configs.
                merged = copy.deepcopy(base)
                deep_update(merged, scenario)

                # MLflow experiment selection:
                # - If --mlflow-experiment is provided, always use it.
                # - Else, if --scenario-name was explicitly provided, use it as experiment name
                #   so few-shot sweeps land in their own experiment (e.g. full_ft_hv_fewshot).
                if args.mlflow_experiment:
                    _set_nested(merged, ("mlflow", "experiment"), str(args.mlflow_experiment))
                elif scenario_name_cli is not None:
                    _set_nested(merged, ("mlflow", "experiment"), str(scenario_name))

                _set_nested(merged, ("run", "seed"), int(seed))

                # Few-shot budget: subsample only training split.
                # Zero-shot: force test-only mode.
                if float(budget) <= 0.0:
                    _set_nested(merged, ("run", "mode"), "test")
                    # Keep target_budget visible/loggable as 0.0
                    _set_nested(merged, ("split", "train_subset_frac"), 0.0)
                else:
                    _set_nested(merged, ("split", "train_subset_frac"), float(budget))

                if r_value is not None:
                    _set_nested(merged, ("peft", "lora_r"), int(r_value))
                if alpha_value is not None:
                    _set_nested(merged, ("peft", "lora_alpha"), int(alpha_value))

                # Set run.name unless already set or unless forced.
                run_block = merged.get("run")
                run_map = run_block if isinstance(run_block, dict) else {}
                existing_name = run_map.get("name")
                if args.force_run_name or not existing_name:
                    run_name = str(args.name_template).format(
                        scenario=scenario_name,
                        seed=int(seed),
                        budget=f"{float(budget):g}",
                        r=str(r_value) if r_value is not None else "",
                        alpha=str(alpha_value) if alpha_value is not None else "",
                    )
                    _set_nested(merged, ("run", "name"), run_name)

                suffix_parts: list[str] = [f"b{float(budget):g}"]
                if r_value is not None:
                    suffix_parts.append(f"r{r_value}")
                if alpha_value is not None:
                    suffix_parts.append(f"a{alpha_value}")
                suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""

                cfg_path = out_dir / f"seed_{seed}{suffix}.yaml"
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(merged, f, sort_keys=False)
                merged_configs.append((cfg_path, int(seed), float(budget), r_value, alpha_value))

    train_args = list(args.train_args or [])
    # argparse.REMAINDER keeps leading "--" sometimes; normalize.
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]

    log.info("Scenario: %s", scenario_name)
    log.info("Seeds: %s", seeds)
    if has_grid:
        log.info("LoRA grid r=%s, alpha=%s", grid_rs, grid_alphas)
    if budgets != [1.0]:
        log.info("Budgets: %s", budgets)
    log.info("Generated configs: %s", out_dir)
    if train_args:
        log.info("Forwarded args: %s", train_args)

    results: list[tuple[int, float, int]] = []  # (seed, budget, returncode)
    for cfg_path, seed, budget, r_value, alpha_value in merged_configs:
        cmd = [args.python_exe, "-m", "train", "--config", str(cfg_path), *train_args]
        log.info("===")
        log.info(
            "Seed %s (budget=%s, r=%s, alpha=%s): %s",
            seed,
            budget,
            r_value,
            alpha_value,
            " ".join(cmd),
        )
        if args.dry_run:
            results.append((seed, float(budget), 0))
            continue

        env = os.environ.copy()
        proc = subprocess.run(cmd, env=env)
        results.append((seed, float(budget), int(proc.returncode)))

        if proc.returncode != 0 and not args.continue_on_error:
            log.error(
                "Stopping sweep due to failure (seed=%s, code=%s).",
                seed,
                proc.returncode,
            )
            break

    log.info("Summary:")
    ok = 0
    for seed, budget, rc in results:
        status = "OK" if rc == 0 else f"FAIL({rc})"
        if rc == 0:
            log.info("seed=%s budget=%s: %s", seed, budget, status)
        else:
            log.error("seed=%s budget=%s: %s", seed, budget, status)
        if rc == 0:
            ok += 1
    log.info("Done: %s/%s succeeded", ok, len(results))

    # Non-zero if any failed.
    return 0 if all(rc == 0 for _, _, rc in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

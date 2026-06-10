from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from . import dynamic_env
from .comparison import AVAILABLE_ALGORITHMS, build_model_summary, run_algorithm_on_instance
from .exact_solver import solve_tsp_exact
from .utils import generate_cities


REPORT_ROOT = Path("results") / "mega_reports"


@dataclass(frozen=True)
class MegaReportScenario:
    scenario_id: str
    city_count: int
    instance_seed: int | None
    mode_seed: int | None
    coords: np.ndarray
    exact_length: float | None
    exact_status: str | None
    exact_runtime_s: float
    exact_is_optimal: bool
    exact_method: str | None


def _make_report_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]


def _scenario_seed(base_seed: int | None, city_index: int, instance_index: int) -> int | None:
    if base_seed is None:
        return None
    return int(base_seed) + city_index * 1009 + instance_index * 37


def _mode_seed(base_seed: int | None, city_index: int, instance_index: int) -> int | None:
    if base_seed is None:
        return None
    return int(base_seed) + 100000 + city_index * 2003 + instance_index * 53


def _run_exact_baseline(coords: np.ndarray, max_seconds: int, blocked_mask: np.ndarray | None) -> dict[str, Any]:
    started = datetime.utcnow()
    try:
        result = solve_tsp_exact(coords, max_seconds=max_seconds, blocked_mask=blocked_mask)
        result["runtime_s"] = float((datetime.utcnow() - started).total_seconds())
        return result
    except Exception as exc:
        elapsed = float((datetime.utcnow() - started).total_seconds())
        return {
            "tour": None,
            "length": None,
            "is_optimal": False,
            "method": "failed",
            "status": "ERROR",
            "solve_time_s": elapsed,
            "runtime_s": elapsed,
            "error": str(exc),
        }


def _build_scenarios(
    city_sizes: Iterable[int],
    *,
    instances_per_size: int,
    base_seed: int | None,
    clustered: bool,
    tsp_mode: str,
    exact_timeout: int,
    blocked_fraction: float,
    blocked_count: int | None,
    penalty: float,
    auto_relax: bool,
) -> list[MegaReportScenario]:
    scenarios: list[MegaReportScenario] = []
    for city_index, city_count in enumerate(city_sizes):
        for instance_index in range(int(instances_per_size)):
            instance_seed = _scenario_seed(base_seed, city_index, instance_index)
            mode_seed = _mode_seed(base_seed, city_index, instance_index) if tsp_mode != "standard" else None
            coords = generate_cities(int(city_count), seed=instance_seed, clustered=clustered)

            dynamic_env.set_params(
                mode=tsp_mode,
                blocked_fraction=float(blocked_fraction),
                blocked_count=None if blocked_count is None else int(blocked_count),
                penalty=float(penalty),
                seed=mode_seed,
                auto_relax=bool(auto_relax),
            )
            blocked_mask = dynamic_env.get_block_mask(len(coords), iteration=0 if tsp_mode == "dynamic" else None)
            exact = _run_exact_baseline(coords, exact_timeout, blocked_mask)

            scenarios.append(
                MegaReportScenario(
                    scenario_id=f"{int(city_count)}:{instance_index}",
                    city_count=int(city_count),
                    instance_seed=instance_seed,
                    mode_seed=mode_seed,
                    coords=coords,
                    exact_length=None if exact.get("length") is None else float(exact.get("length")),
                    exact_status=str(exact.get("status")) if exact.get("status") is not None else None,
                    exact_runtime_s=float(exact.get("runtime_s", exact.get("solve_time_s", 0.0))),
                    exact_is_optimal=bool(exact.get("is_optimal", False)),
                    exact_method=exact.get("method"),
                )
            )
    return scenarios


def _resolve_config_values(model_label: str, model_config_values: dict[str, Iterable[int]]) -> list[int]:
    values = [int(v) for v in model_config_values.get(model_label, []) if int(v) > 0]
    if values:
        return values
    if any(spec.label == model_label for spec in AVAILABLE_ALGORITHMS):
        return [10, 20, 40]
    return [10]


def _algorithm_tasks_for_scenario(
    scenario: MegaReportScenario,
    *,
    selected_models: Iterable[str],
    model_config_values: dict[str, Iterable[int]],
    iterations: int,
    target_gap_pct: float,
    aco_settings: dict[str, Any],
    pso_settings: dict[str, Any],
    abc_settings: dict[str, Any],
    ga_settings: dict[str, Any],
    tsp_mode: str,
) -> list[dict[str, Any]]:
    exact_length = scenario.exact_length if scenario.exact_status == "OPTIMAL" else None
    tasks: list[dict[str, Any]] = []
    for model_label in selected_models:
        for config_value in _resolve_config_values(model_label, model_config_values):
            tasks.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "city_count": scenario.city_count,
                    "city_seed": scenario.instance_seed,
                    "mode_seed": scenario.mode_seed,
                    "tsp_mode": tsp_mode,
                    "coords": scenario.coords,
                    "model_label": model_label,
                    "config_value": int(config_value),
                    "iterations": int(iterations),
                    "exact_length": exact_length,
                    "exact_status": scenario.exact_status,
                    "target_gap_pct": float(target_gap_pct),
                    "aco_settings": dict(aco_settings),
                    "pso_settings": dict(pso_settings),
                    "abc_settings": dict(abc_settings),
                    "ga_settings": dict(ga_settings),
                }
            )
    return tasks


def _run_mega_task(task: dict[str, Any]) -> dict[str, Any]:
    dynamic_env.set_params(
        mode=str(task["tsp_mode"]),
        blocked_fraction=float(dynamic_env.BLOCKED_FRACTION),
        blocked_count=dynamic_env.BLOCKED_COUNT,
        penalty=float(dynamic_env.PENALTY),
        seed=task.get("mode_seed"),
        auto_relax=bool(dynamic_env.AUTO_RELAX),
    )
    row = run_algorithm_on_instance(
        str(task["model_label"]),
        np.asarray(task["coords"], dtype=float),
        int(task["config_value"]),
        iterations=int(task["iterations"]),
        seed=None if task.get("city_seed") is None else int(task["city_seed"]),
        exact_length=task.get("exact_length"),
        exact_status=task.get("exact_status"),
        aco_settings=task.get("aco_settings", {}),
        pso_settings=task.get("pso_settings", {}),
        abc_settings=task.get("abc_settings", {}),
        ga_settings=task.get("ga_settings", {}),
        target_gap_pct=float(task.get("target_gap_pct", 0.0)),
    )
    row.update(
        {
            "scenario_id": task["scenario_id"],
            "tsp_mode": task["tsp_mode"],
            "city_count": int(task["city_count"]),
            "city_seed": task.get("city_seed"),
            "mode_seed": task.get("mode_seed"),
        }
    )
    return row


def _row_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    for column in [
        "best_len",
        "optimality_gap_pct",
        "convergence_time_s",
        "objective_evals_to_convergence",
        "objective_evals_total",
        "run_time_s",
        "exact_length",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def build_mega_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = _row_dataframe(rows)
    if frame.empty:
        return []
    complete = frame[(frame["status"] == "complete") & frame["best_len"].notna()].copy()
    if complete.empty:
        return []
    summaries: list[dict[str, Any]] = []
    for model_label, group in complete.groupby("model"):
        optimal = group.get("optimal_match", pd.Series(dtype=bool)).fillna(False)
        stopped = group.get("stopped_early", pd.Series(dtype=bool)).fillna(False)
        summaries.append(
            {
                "model": model_label,
                "avg_gap_pct": float(group["optimality_gap_pct"].fillna(0.0).mean()),
                "avg_convergence_time_s": float(group["convergence_time_s"].fillna(0.0).mean()),
                "avg_evals_to_convergence": float(group["objective_evals_to_convergence"].fillna(0.0).mean()),
                "avg_run_time_s": float(group["run_time_s"].fillna(0.0).mean()),
                "optimal_hits": int(optimal.sum()),
                "exact_match_rate": float(optimal.mean()) if len(optimal) else 0.0,
                "early_stop_rate": float(stopped.mean()) if len(stopped) else 0.0,
                "runs": int(len(group)),
            }
        )
    return summaries


def _city_summary(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = _row_dataframe(rows)
    if frame.empty:
        return frame
    complete = frame[(frame["status"] == "complete") & frame["best_len"].notna()].copy()
    if complete.empty:
        return complete
    return (
        complete.groupby(["city_count", "model"], as_index=False)
        .agg(
            avg_gap_pct=("optimality_gap_pct", "mean"),
            avg_run_time_s=("run_time_s", "mean"),
            avg_convergence_time_s=("convergence_time_s", "mean"),
            avg_evals_to_convergence=("objective_evals_to_convergence", "mean"),
            exact_match_rate=("optimal_match", "mean"),
            runs=("model", "count"),
        )
        .sort_values(["city_count", "model"])
    )


def _build_insights(report: dict[str, Any]) -> list[str]:
    summary = report.get("summary") or []
    if not summary:
        return ["No complete runs were produced."]
    summary_sorted = sorted(summary, key=lambda row: row["model"])
    best_gap = min(summary_sorted, key=lambda row: row["avg_gap_pct"])
    best_time = min(summary_sorted, key=lambda row: row["avg_run_time_s"])
    best_evals = min(summary_sorted, key=lambda row: row["avg_evals_to_convergence"])
    best_match = max(summary_sorted, key=lambda row: row["exact_match_rate"])
    insights = [
        f"Best average route quality: {best_gap['model']} with an average gap of {best_gap['avg_gap_pct']:.3f}%.",
        f"Fastest end-to-end runtime: {best_time['model']} with {best_time['avg_run_time_s']:.4f}s on average.",
        f"Lowest effort to convergence: {best_evals['model']} with {best_evals['avg_evals_to_convergence']:.1f} evaluations on average.",
        f"Highest exact-match rate: {best_match['model']} at {best_match['exact_match_rate'] * 100.0:.1f}%.",
    ]
    if str(report.get("variant")) == "dynamic":
        insights.append("Dynamic mode is summarized from the initial mask baseline plus iteration-level adaptation statistics.")
    if str(report.get("variant")) == "noisy":
        insights.append("Noisy mode uses a fixed blocked-edge mask per instance, so comparisons stay stable across algorithms.")
    return insights


def _make_pdf(report: dict[str, Any], pdf_path: Path) -> None:
    rows = report.get("rows", [])
    summary = pd.DataFrame(report.get("summary", []))
    city_summary = _city_summary(rows)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis("off")
        fig.text(0.06, 0.95, f"Mega TSP Report: {report['variant'].title()}", fontsize=20, fontweight="bold")
        fig.text(0.06, 0.91, f"Created: {report['created_at']}", fontsize=10)
        fig.text(0.06, 0.87, f"Scenarios: {report['settings']['scenario_count']} | Tasks: {report['settings']['task_count']} | Parallel workers: {report['settings']['parallel_workers']}", fontsize=10)
        fig.text(0.06, 0.83, f"City sizes: {', '.join(map(str, report['settings']['city_sizes']))}", fontsize=10)
        fig.text(0.06, 0.79, f"Instances per size: {report['settings']['instances_per_size']} | Iterations: {report['settings']['iterations']} | Exact timeout: {report['settings']['exact_timeout']}s", fontsize=10)
        y = 0.72
        for line in report.get("insights", [])[:8]:
            fig.text(0.06, y, f"• {line}", fontsize=9)
            y -= 0.045
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if not summary.empty:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            axes = axes.flatten()
            summary_sorted = summary.sort_values("model")
            axes[0].bar(summary_sorted["model"], summary_sorted["avg_gap_pct"], color="#f97316")
            axes[0].set_title("Average gap %")
            axes[0].tick_params(axis="x", rotation=20)

            axes[1].bar(summary_sorted["model"], summary_sorted["avg_run_time_s"], color="#38bdf8")
            axes[1].set_title("Average runtime (s)")
            axes[1].tick_params(axis="x", rotation=20)

            axes[2].bar(summary_sorted["model"], summary_sorted["avg_evals_to_convergence"], color="#a78bfa")
            axes[2].set_title("Average evals to convergence")
            axes[2].tick_params(axis="x", rotation=20)

            axes[3].bar(summary_sorted["model"], summary_sorted["exact_match_rate"] * 100.0, color="#34d399")
            axes[3].set_title("Exact-match rate (%)")
            axes[3].tick_params(axis="x", rotation=20)

            fig.suptitle("Model Summary", fontsize=18, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if not city_summary.empty:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            for model_label, group in city_summary.groupby("model"):
                ax.plot(group["city_count"], group["avg_gap_pct"], marker="o", label=model_label)
            ax.set_title("Average gap by city size")
            ax.set_xlabel("City count")
            ax.set_ylabel("Gap %")
            ax.legend()
            ax.grid(True, alpha=0.25)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def run_mega_report(
    *,
    tsp_mode: str,
    city_sizes: Iterable[int],
    instances_per_size: int,
    base_seed: int | None,
    clustered: bool,
    iterations: int,
    exact_timeout: int,
    selected_models: Iterable[str],
    model_config_values: dict[str, Iterable[int]],
    aco_settings: dict[str, Any],
    pso_settings: dict[str, Any],
    abc_settings: dict[str, Any],
    ga_settings: dict[str, Any],
    blocked_fraction: float = 0.05,
    blocked_count: int | None = None,
    penalty: float = 1e6,
    auto_relax: bool = False,
    target_gap_pct: float = 0.0,
    parallel_workers: int = 4,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    report_id = _make_report_id()
    report_dir = REPORT_ROOT / report_id
    report_dir.mkdir(parents=True, exist_ok=True)

    selected_models = [str(label) for label in selected_models] or [spec.label for spec in AVAILABLE_ALGORITHMS if spec.runnable]
    city_sizes = [int(value) for value in city_sizes]
    instances_per_size = int(instances_per_size)
    scenarios = _build_scenarios(
        city_sizes,
        instances_per_size=instances_per_size,
        base_seed=base_seed,
        clustered=clustered,
        tsp_mode=tsp_mode,
        exact_timeout=exact_timeout,
        blocked_fraction=blocked_fraction,
        blocked_count=blocked_count,
        penalty=penalty,
        auto_relax=auto_relax,
    )

    tasks: list[dict[str, Any]] = []
    for scenario in scenarios:
        tasks.extend(
            _algorithm_tasks_for_scenario(
                scenario,
                selected_models=selected_models,
                model_config_values=model_config_values,
                iterations=iterations,
                target_gap_pct=target_gap_pct,
                aco_settings=aco_settings,
                pso_settings=pso_settings,
                abc_settings=abc_settings,
                ga_settings=ga_settings,
                tsp_mode=tsp_mode,
            )
        )

    rows: list[dict[str, Any]] = []
    total_work = max(1, len(scenarios) + len(tasks))
    completed_work = 0
    started = datetime.utcnow()

    def emit(event: dict[str, Any]) -> None:
        if progress_callback is None:
            return
        elapsed = max(1e-9, (datetime.utcnow() - started).total_seconds())
        eta_s = None
        if completed_work > 0:
            eta_s = elapsed / completed_work * max(0, total_work - completed_work)
        progress_callback(
            {
                "type": event.get("type", "progress"),
                "completed_work": completed_work,
                "total_work": total_work,
                "progress": completed_work / float(total_work),
                "eta_s": eta_s,
                **event,
            }
        )

    emit({"type": "prep_started", "scenario_count": len(scenarios), "task_count": len(tasks)})
    for index, scenario in enumerate(scenarios, start=1):
        completed_work += 1
        emit(
            {
                "type": "scenario_prepared",
                "scenario_index": index,
                "scenario_total": len(scenarios),
                "scenario_id": scenario.scenario_id,
                "city_count": scenario.city_count,
            }
        )

    if parallel_workers <= 1 or len(tasks) <= 1:
        for index, task in enumerate(tasks, start=1):
            rows.append(_run_mega_task(task))
            completed_work += 1
            emit(
                {
                    "type": "task_completed",
                    "task_index": index,
                    "task_total": len(tasks),
                    "model": task["model_label"],
                    "scenario_id": task["scenario_id"],
                }
            )
    else:
        with ProcessPoolExecutor(max_workers=int(parallel_workers)) as executor:
            future_map = {executor.submit(_run_mega_task, task): task for task in tasks}
            for index, future in enumerate(as_completed(future_map), start=1):
                task = future_map[future]
                rows.append(future.result())
                completed_work += 1
                emit(
                    {
                        "type": "task_completed",
                        "task_index": index,
                        "task_total": len(tasks),
                        "model": task["model_label"],
                        "scenario_id": task["scenario_id"],
                    }
                )

    summary = build_mega_summary(rows)
    report = {
        "report_id": report_id,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "variant": tsp_mode,
        "settings": {
            "tsp_mode": tsp_mode,
            "city_sizes": city_sizes,
            "instances_per_size": instances_per_size,
            "base_seed": base_seed,
            "clustered": bool(clustered),
            "iterations": int(iterations),
            "exact_timeout": int(exact_timeout),
            "parallel_workers": int(parallel_workers),
            "target_gap_pct": float(target_gap_pct),
            "scenario_count": len(scenarios),
            "task_count": len(tasks),
            "selected_models": selected_models,
            "model_config_values": {key: [int(v) for v in value] for key, value in model_config_values.items()},
            "blocked_fraction": float(blocked_fraction),
            "blocked_count": None if blocked_count is None else int(blocked_count),
            "penalty": float(penalty),
            "auto_relax": bool(auto_relax),
        },
        "scenarios": [
            {
                "scenario_id": scenario.scenario_id,
                "city_count": scenario.city_count,
                "instance_seed": scenario.instance_seed,
                "mode_seed": scenario.mode_seed,
                "exact_length": scenario.exact_length,
                "exact_status": scenario.exact_status,
                "exact_runtime_s": scenario.exact_runtime_s,
                "exact_is_optimal": scenario.exact_is_optimal,
                "exact_method": scenario.exact_method,
            }
            for scenario in scenarios
        ],
        "rows": rows,
        "summary": summary,
    }
    report["insights"] = _build_insights(report)

    json_path = report_dir / "report.json"
    pdf_path = report_dir / "report.pdf"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    _make_pdf(report, pdf_path)

    report.update(
        {
            "files": {
                "json": str(json_path),
                "pdf": str(pdf_path),
            },
            "elapsed_s": float((datetime.utcnow() - started).total_seconds()),
        }
    )
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def load_report_json(report_path: str | Path) -> dict[str, Any]:
    with open(Path(report_path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def latest_report_path() -> Path | None:
    if not REPORT_ROOT.exists():
        return None
    candidates = sorted(REPORT_ROOT.glob("*/report.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def report_summary_frame(report: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(report.get("summary", []))


def report_rows_frame(report: dict[str, Any]) -> pd.DataFrame:
    return _row_dataframe(report.get("rows", []))


def city_summary_frame(report: dict[str, Any]) -> pd.DataFrame:
    return _city_summary(report.get("rows", []))
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable
from uuid import uuid4

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from . import dynamic_env
from .comparison import AVAILABLE_ALGORITHMS, build_model_summary, run_algorithm_on_instance, run_exact_baseline
from .utils import generate_cities


REPORT_ROOT = Path("results") / "mega_reports"


@dataclass(frozen=True)
class MegaReportScenario:
    scenario_id: str
    city_count: int
    instance_seed: int | None
    mode_seed: int | None
    clustered: bool
    coords: np.ndarray
    exact_length: float | None
    exact_status: str | None
    exact_runtime_s: float
    exact_is_optimal: bool
    exact_method: str | None


def _make_report_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]


def _scenario_seed(base_seed: int | None, city_index: int, instance_index: int) -> int | None:
    if base_seed is None:
        return None
    return int(base_seed) + city_index * 1009 + instance_index * 37


def _mode_seed(base_seed: int | None, city_index: int, instance_index: int) -> int | None:
    if base_seed is None:
        return None
    return int(base_seed) + 100000 + city_index * 2003 + instance_index * 53


def _build_scenarios(
    city_sizes: Iterable[int],
    *,
    instances_per_size: int,
    base_seed: int | None,
    clustered: bool,
    tsp_mode: str,
    exact_timeout: int,
) -> list[MegaReportScenario]:
    scenarios: list[MegaReportScenario] = []
    for city_index, city_count in enumerate(city_sizes):
        for instance_index in range(int(instances_per_size)):
            instance_seed = _scenario_seed(base_seed, city_index, instance_index)
            mode_seed = _mode_seed(base_seed, city_index, instance_index) if tsp_mode != "standard" else None
            coords = generate_cities(int(city_count), seed=instance_seed, clustered=clustered)
            dynamic_env.set_params(
                mode=tsp_mode,
                blocked_fraction=float(dynamic_env.BLOCKED_FRACTION),
                blocked_count=dynamic_env.BLOCKED_COUNT,
                penalty=float(dynamic_env.PENALTY),
                seed=mode_seed,
                auto_relax=bool(dynamic_env.AUTO_RELAX),
            )
            blocked_mask = dynamic_env.get_block_mask(len(coords), iteration=0 if tsp_mode == "dynamic" else None)
            exact = run_exact_baseline(coords, exact_timeout)
            if tsp_mode == "dynamic" and exact.get("status") == "OPTIMAL":
                # The benchmark uses the iteration-0 mask as the baseline for dynamic runs.
                exact.setdefault("baseline_iteration", 0)
            scenarios.append(
                MegaReportScenario(
                    scenario_id=f"{int(city_count)}:{instance_index}",
                    city_count=int(city_count),
                    instance_seed=instance_seed,
                    mode_seed=mode_seed,
                    clustered=bool(clustered),
                    coords=coords,
                    exact_length=None if exact.get("length") is None else float(exact.get("length")),
                    exact_status=str(exact.get("status")) if exact.get("status") is not None else None,
                    exact_runtime_s=float(exact.get("runtime_s", exact.get("solve_time_s", 0.0))),
                    exact_is_optimal=bool(exact.get("is_optimal", False)),
                    exact_method=exact.get("method"),
                )
            )
    return scenarios


def _algorithm_tasks_for_scenario(
    scenario: MegaReportScenario,
    *,
    selected_models: Iterable[str],
    model_config_values: dict[str, Iterable[int]],
    iterations: int,
    target_gap_pct: float,
    aco_settings: dict[str, Any],
    pso_settings: dict[str, Any],
    abc_settings: dict[str, Any],
    ga_settings: dict[str, Any],
    tsp_mode: str,
) -> list[dict[str, Any]]:
    exact_length = scenario.exact_length if scenario.exact_status == "OPTIMAL" else None
    tasks: list[dict[str, Any]] = []
    for model_label in selected_models:
        for config_value in model_config_values.get(model_label, []):
            tasks.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "city_count": scenario.city_count,
                    "city_seed": scenario.instance_seed,
                    "mode_seed": scenario.mode_seed,
                    "tsp_mode": tsp_mode,
                    "coords": scenario.coords,
                    "model_label": model_label,
                    "config_value": int(config_value),
                    "iterations": int(iterations),
                    "exact_length": exact_length,
                    "exact_status": scenario.exact_status,
                    "target_gap_pct": float(target_gap_pct),
                    "aco_settings": dict(aco_settings),
                    "pso_settings": dict(pso_settings),
                    "abc_settings": dict(abc_settings),
                    "ga_settings": dict(ga_settings),
                }
            )
    return tasks


def _run_mega_task(task: dict[str, Any]) -> dict[str, Any]:
    dynamic_env.set_params(
        mode=str(task["tsp_mode"]),
        blocked_fraction=float(dynamic_env.BLOCKED_FRACTION),
        blocked_count=dynamic_env.BLOCKED_COUNT,
        penalty=float(dynamic_env.PENALTY),
        seed=task.get("mode_seed"),
        auto_relax=bool(dynamic_env.AUTO_RELAX),
    )
    row = run_algorithm_on_instance(
        str(task["model_label"]),
        task["coords"],
        int(task["config_value"]),
        iterations=int(task["iterations"]),
        seed=None if task.get("city_seed") is None else int(task["city_seed"]),
        exact_length=task.get("exact_length"),
        exact_status=task.get("exact_status"),
        aco_settings=task.get("aco_settings", {}),
        pso_settings=task.get("pso_settings", {}),
        abc_settings=task.get("abc_settings", {}),
        ga_settings=task.get("ga_settings", {}),
        target_gap_pct=float(task.get("target_gap_pct", 0.0)),
    )
    row.update(
        {
            "scenario_id": task["scenario_id"],
            "tsp_mode": task["tsp_mode"],
            "city_count": int(task["city_count"]),
            "city_seed": task.get("city_seed"),
            "mode_seed": task.get("mode_seed"),
        }
    )
    return row


def _row_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for column in ["best_len", "optimality_gap_pct", "convergence_time_s", "objective_evals_to_convergence", "objective_evals_total", "run_time_s", "exact_length"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _safe_float_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns or df.empty:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def build_mega_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = _row_dataframe(rows)
    if df.empty:
        return []
    complete = df[(df.get("status") == "complete") & df["best_len"].notna()].copy()
    if complete.empty:
        return []
    summaries: list[dict[str, Any]] = []
    for model_label, group in complete.groupby("model"):
        summaries.append(
            {
                "model": model_label,
                "avg_gap_pct": float(group["optimality_gap_pct"].fillna(0.0).mean()),
                "avg_convergence_time_s": float(group["convergence_time_s"].fillna(0.0).mean()),
                "avg_evals_to_convergence": float(group["objective_evals_to_convergence"].fillna(0.0).mean()),
                "avg_run_time_s": float(group["run_time_s"].fillna(0.0).mean()),
                "optimal_hits": int(group.get("optimal_match", pd.Series(dtype=bool)).fillna(False).sum()),
                "exact_match_rate": float(group.get("optimal_match", pd.Series(dtype=bool)).fillna(False).mean()),
                "early_stop_rate": float(group.get("stopped_early", pd.Series(dtype=bool)).fillna(False).mean()),
                "runs": int(len(group)),
            }
        )
    return summaries


def _aggregate_by_city(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = _row_dataframe(rows)
    if df.empty:
        return df
    complete = df[(df.get("status") == "complete") & df["best_len"].notna()].copy()
    if complete.empty:
        return complete
    return (
        complete.groupby(["city_count", "model"], as_index=False)
        .agg(
            avg_gap_pct=("optimality_gap_pct", "mean"),
            avg_run_time_s=("run_time_s", "mean"),
            avg_convergence_time_s=("convergence_time_s", "mean"),
            avg_evals_to_convergence=("objective_evals_to_convergence", "mean"),
            exact_match_rate=("optimal_match", "mean"),
            runs=("model", "count"),
        )
        .sort_values(["city_count", "model"])
    )


def _winner_counts(rows: list[dict[str, Any]], key: str) -> pd.DataFrame:
    df = _row_dataframe(rows)
    if df.empty:
        return pd.DataFrame(columns=["model", "wins"])
    complete = df[(df.get("status") == "complete") & df[key].notna()].copy()
    if complete.empty:
        return pd.DataFrame(columns=["model", "wins"])
    winners = []
    for _, group in complete.groupby("scenario_id"):
        if key in ("run_time_s", "convergence_time_s", "optimality_gap_pct", "objective_evals_to_convergence"):
            best_value = group[key].min()
        else:
            best_value = group[key].max()
        best_rows = group[group[key] == best_value]
        for model in best_rows["model"].tolist():
            winners.append(model)
    if not winners:
        return pd.DataFrame(columns=["model", "wins"])
    return pd.DataFrame({"model": winners}).value_counts().reset_index(name="wins")


def _make_pdf(report: dict[str, Any], pdf_path: Path) -> None:
    rows = report.get("rows", [])
    df = _row_dataframe(rows)
    summary = pd.DataFrame(report.get("summary", []))
    city_summary = _aggregate_by_city(rows)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis("off")
        title = f"Mega TSP Report: {report['settings']['tsp_mode']}"
        fig.text(0.06, 0.95, title, fontsize=20, fontweight="bold")
        fig.text(0.06, 0.91, f"Created: {report['created_at']}", fontsize=10)
        fig.text(0.06, 0.87, f"Scenarios: {report['settings']['scenario_count']} | Tasks: {report['settings']['task_count']} | Parallel workers: {report['settings']['parallel_workers']}", fontsize=10)
        fig.text(0.06, 0.83, f"City sizes: {', '.join(map(str, report['settings']['city_sizes']))}", fontsize=10)
        fig.text(0.06, 0.79, f"Instances per size: {report['settings']['instances_per_size']} | Iterations: {report['settings']['iterations']} | Exact timeout: {report['settings']['exact_timeout']}s", fontsize=10)
        y = 0.72
        for line in report.get("insights", [])[:8]:
            fig.text(0.06, y, f"• {line}", fontsize=9)
            y -= 0.045
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if not summary.empty:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            axes = axes.flatten()
            summary_sorted = summary.sort_values("model")
            axes[0].bar(summary_sorted["model"], summary_sorted["avg_gap_pct"], color="#f97316")
            axes[0].set_title("Average gap %")
            axes[0].tick_params(axis="x", rotation=20)

            axes[1].bar(summary_sorted["model"], summary_sorted["avg_run_time_s"], color="#38bdf8")
            axes[1].set_title("Average runtime (s)")
            axes[1].tick_params(axis="x", rotation=20)

            axes[2].bar(summary_sorted["model"], summary_sorted["avg_evals_to_convergence"], color="#a78bfa")
            axes[2].set_title("Average evals to convergence")
            axes[2].tick_params(axis="x", rotation=20)

            axes[3].bar(summary_sorted["model"], summary_sorted["exact_match_rate"] * 100.0, color="#34d399")
            axes[3].set_title("Exact-match rate (%)")
            axes[3].tick_params(axis="x", rotation=20)

            fig.suptitle("Model Summary", fontsize=18, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if not city_summary.empty:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            for model_label, group in city_summary.groupby("model"):
                ax.plot(group["city_count"], group["avg_gap_pct"], marker="o", label=model_label)
            ax.set_title("Average gap by city size")
            ax.set_xlabel("City count")
            ax.set_ylabel("Gap %")
            ax.legend()
            ax.grid(True, alpha=0.25)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _build_insights(report_rows: list[dict[str, Any]]) -> list[str]:
    summary = build_model_summary(report_rows)
    if not summary:
        return ["No complete rows were produced."]
    summary = sorted(summary, key=lambda row: row["model"])
    best_gap = min(summary, key=lambda row: row["avg_gap_pct"])
    best_time = min(summary, key=lambda row: row["avg_run_time_s"])
    best_evals = min(summary, key=lambda row: row["avg_evals_to_convergence"])
    best_match = max(summary, key=lambda row: row["exact_match_rate"])
    best_early = max(summary, key=lambda row: row["runs"])
    insights = [
        f"Best average route quality: {best_gap['model']} with an average gap of {best_gap['avg_gap_pct']:.3f}%.",
        f"Fastest end-to-end runtime: {best_time['model']} with {best_time['avg_run_time_s']:.4f}s on average.",
        f"Lowest effort to convergence: {best_evals['model']} with {best_evals['avg_evals_to_convergence']:.1f} evaluations on average.",
        f"Highest exact-match rate: {best_match['model']} at {best_match['exact_match_rate'] * 100.0:.1f}%.",
    ]
    if best_early:
        insights.append(
            "Early-stop behavior is preserved in standard/noisy variants; dynamic mode is reported separately because the environment changes during the run."
        )
    return insights


def run_mega_report(
    *,
    tsp_mode: str,
    city_sizes: Iterable[int],
    instances_per_size: int,
    base_seed: int | None,
    clustered: bool,
    iterations: int,
    exact_timeout: int,
    selected_models: Iterable[str],
    model_config_values: dict[str, Iterable[int]],
    aco_settings: dict[str, Any],
    pso_settings: dict[str, Any],
    abc_settings: dict[str, Any],
    ga_settings: dict[str, Any],
    target_gap_pct: float = 0.0,
    parallel_workers: int = 4,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    report_id = _make_report_id()
    report_dir = REPORT_ROOT / report_id
    report_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _build_scenarios(
        city_sizes,
        instances_per_size=instances_per_size,
        base_seed=base_seed,
        clustered=clustered,
        tsp_mode=tsp_mode,
        exact_timeout=exact_timeout,
    )
    tasks: list[dict[str, Any]] = []
    for scenario in scenarios:
        tasks.extend(
            _algorithm_tasks_for_scenario(
                scenario,
                selected_models=selected_models,
                model_config_values=model_config_values,
                iterations=iterations,
                target_gap_pct=target_gap_pct,
                aco_settings=aco_settings,
                pso_settings=pso_settings,
                abc_settings=abc_settings,
                ga_settings=ga_settings,
                tsp_mode=tsp_mode,
            )
        )

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    total_work = max(1, len(scenarios) + len(tasks))
    completed_work = 0

    def emit(event: dict[str, Any]) -> None:
        if progress_callback is not None:
            elapsed = max(1e-9, time.perf_counter() - started)
            eta_s = None
            if completed_work > 0:
                eta_s = elapsed / completed_work * max(0, total_work - completed_work)
            progress_callback(
                {
                    "type": event.get("type", "progress"),
                    "completed_work": completed_work,
                    "total_work": total_work,
                    "progress": completed_work / float(total_work),
                    "eta_s": eta_s,
                    **event,
                }
            )

    emit({"type": "prep_started", "scenario_count": len(scenarios), "task_count": len(tasks)})
    for index, scenario in enumerate(scenarios, start=1):
        completed_work += 1
        emit(
            {
                "type": "scenario_prepared",
                "scenario_index": index,
                "scenario_total": len(scenarios),
                "scenario_id": scenario.scenario_id,
                "city_count": scenario.city_count,
            }
        )

    if parallel_workers <= 1 or len(tasks) <= 1:
        for index, task in enumerate(tasks, start=1):
            row = _run_mega_task(task)
            rows.append(row)
            completed_work += 1
            emit(
                {
                    "type": "task_completed",
                    "task_index": index,
                    "task_total": len(tasks),
                    "model": task["model_label"],
                    "scenario_id": task["scenario_id"],
                }
            )
    else:
        with ProcessPoolExecutor(max_workers=int(parallel_workers)) as executor:
            future_map = {executor.submit(_run_mega_task, task): task for task in tasks}
            for index, future in enumerate(as_completed(future_map), start=1):
                task = future_map[future]
                row = future.result()
                rows.append(row)
                completed_work += 1
                emit(
                    {
                        "type": "task_completed",
                        "task_index": index,
                        "task_total": len(tasks),
                        "model": task["model_label"],
                        "scenario_id": task["scenario_id"],
                    }
                )

    summary = build_model_summary(rows)
    insights = _build_insights(rows)
    report = {
        "report_id": report_id,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "settings": {
            "tsp_mode": tsp_mode,
            "city_sizes": [int(value) for value in city_sizes],
            "instances_per_size": int(instances_per_size),
            "base_seed": base_seed,
            "clustered": bool(clustered),
            "iterations": int(iterations),
            "exact_timeout": int(exact_timeout),
            "parallel_workers": int(parallel_workers),
            "target_gap_pct": float(target_gap_pct),
            "scenario_count": len(scenarios),
            "task_count": len(tasks),
            "selected_models": list(selected_models),
            "model_config_values": {key: [int(v) for v in value] for key, value in model_config_values.items()},
        },
        "scenarios": [
            {
                "scenario_id": scenario.scenario_id,
                "city_count": scenario.city_count,
                "instance_seed": scenario.instance_seed,
                "mode_seed": scenario.mode_seed,
                "exact_length": scenario.exact_length,
                "exact_status": scenario.exact_status,
                "exact_runtime_s": scenario.exact_runtime_s,
                "exact_is_optimal": scenario.exact_is_optimal,
                "exact_method": scenario.exact_method,
            }
            for scenario in scenarios
        ],
        "rows": rows,
        "summary": summary,
        "insights": insights,
    }

    json_path = report_dir / "report.json"
    pdf_path = report_dir / "report.pdf"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    _make_pdf(report, pdf_path)

    report.update(
        {
            "files": {
                "json": str(json_path),
                "pdf": str(pdf_path),
            },
            "elapsed_s": float(time.perf_counter() - started),
        }
    )
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def load_report_json(report_path: str | Path) -> dict[str, Any]:
    with open(Path(report_path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def latest_report_path() -> Path | None:
    if not REPORT_ROOT.exists():
        return None
    candidates = sorted(REPORT_ROOT.glob("*/report.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def report_summary_frame(report: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(report.get("summary", []))


def report_rows_frame(report: dict[str, Any]) -> pd.DataFrame:
    return _row_dataframe(report.get("rows", []))


def city_summary_frame(report: dict[str, Any]) -> pd.DataFrame:
    return _aggregate_by_city(report.get("rows", []))
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import math
import time
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from . import dynamic_env
from .comparison import AVAILABLE_ALGORITHMS, run_algorithm_on_instance, run_exact_baseline
from .utils import generate_cities


RESULTS_DIR = Path("results") / "mega_reports"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _safe_float(value: Any, fallback: float | None = None) -> float | None:
    try:
        if value is None:
            return fallback
        if isinstance(value, bool):
            return float(int(value))
        return float(value)
    except Exception:
        return fallback


def _safe_int(value: Any, fallback: int | None = None) -> int | None:
    try:
        if value is None:
            return fallback
        return int(value)
    except Exception:
        return fallback


def _resolve_config_values(model_label: str, model_config_values: dict[str, list[int]]) -> list[int]:
    values = [int(v) for v in model_config_values.get(model_label, []) if int(v) > 0]
    if values:
        return values
    spec = next((item for item in AVAILABLE_ALGORITHMS if item.label == model_label), None)
    if spec is None:
        return [10]
    return [10, 20, 40]


def _build_instances(city_sizes: Iterable[int], seeds: Iterable[int], clustered: bool) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for city_count in city_sizes:
        for seed in seeds:
            instance_seed = int(seed)
            coords = generate_cities(int(city_count), seed=instance_seed, clustered=bool(clustered))
            instances.append(
                {
                    "city_count": int(city_count),
                    "seed": instance_seed,
                    "coords": coords,
                }
            )
    return instances


def _variant_settings(variant: str, seed: int | None, blocked_fraction: float, blocked_count: int | None, penalty: float, auto_relax: bool) -> None:
    dynamic_env.set_params(
        mode=variant,
        blocked_fraction=float(blocked_fraction),
        blocked_count=None if blocked_count is None else int(blocked_count),
        penalty=float(penalty),
        seed=None if seed is None else int(seed),
        auto_relax=bool(auto_relax),
    )


def _exact_baseline_for_variant(coords: np.ndarray, *, variant: str, exact_timeout: int, blocked_fraction: float, blocked_count: int | None, penalty: float, seed: int | None, auto_relax: bool) -> dict[str, Any]:
    _variant_settings(variant, seed=seed, blocked_fraction=blocked_fraction, blocked_count=blocked_count, penalty=penalty, auto_relax=auto_relax)
    blocked_mask = dynamic_env.get_block_mask(len(coords), iteration=0 if variant == "dynamic" else None)
    return run_exact_baseline(coords, exact_timeout) if blocked_mask is None else run_exact_baseline_with_mask(coords, exact_timeout, blocked_mask)


def run_exact_baseline_with_mask(coords: np.ndarray, max_seconds: int, blocked_mask: np.ndarray | None) -> dict[str, Any]:
    from .exact_solver import solve_tsp_exact

    started = time.perf_counter()
    try:
        result = solve_tsp_exact(coords, max_seconds=max_seconds, blocked_mask=blocked_mask)
        result["runtime_s"] = float(time.perf_counter() - started)
        return result
    except Exception as exc:
        return {
            "tour": None,
            "length": None,
            "is_optimal": False,
            "method": "failed",
            "status": "ERROR",
            "solve_time_s": float(time.perf_counter() - started),
            "runtime_s": float(time.perf_counter() - started),
            "error": str(exc),
        }


def _task_worker(task: dict[str, Any]) -> dict[str, Any]:
    variant = str(task["variant"])
    _variant_settings(
        variant,
        seed=_safe_int(task.get("noise_seed")),
        blocked_fraction=float(task.get("blocked_fraction", 0.05)),
        blocked_count=_safe_int(task.get("blocked_count")),
        penalty=float(task.get("penalty", 1e6)),
        auto_relax=bool(task.get("auto_relax", False)),
    )
    coords = np.asarray(task["coords"], dtype=float)
    row = run_algorithm_on_instance(
        str(task["model_label"]),
        coords,
        int(task["config_value"]),
        iterations=int(task["iterations"]),
        seed=_safe_int(task.get("run_seed")),
        exact_length=_safe_float(task.get("exact_length")),
        exact_status=str(task.get("exact_status")) if task.get("exact_status") is not None else None,
        aco_settings=dict(task.get("aco_settings", {})),
        pso_settings=dict(task.get("pso_settings", {})),
        abc_settings=dict(task.get("abc_settings", {})),
        ga_settings=dict(task.get("ga_settings", {})),
        target_gap_pct=float(task.get("target_gap_pct", 0.0)),
    )
    row.update(
        {
            "variant": variant,
            "instance_seed": _safe_int(task.get("instance_seed")),
            "noise_seed": _safe_int(task.get("noise_seed")),
            "blocked_fraction": float(task.get("blocked_fraction", 0.05)),
            "blocked_count": _safe_int(task.get("blocked_count")),
            "parallel_worker": bool(task.get("parallel_worker", False)),
        }
    )
    return row


def _aggregate_frame(rows: list[dict[str, Any]], group_cols: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    complete = frame[(frame["status"] == "complete") & frame["best_len"].notna()].copy()
    if complete.empty:
        return pd.DataFrame()
    aggregations = {
        "best_len": "mean",
        "optimality_gap_pct": "mean",
        "convergence_time_s": "mean",
        "objective_evals_to_convergence": "mean",
        "objective_evals_total": "mean",
        "run_time_s": "mean",
        "stopped_early": "mean",
        "optimal_match": "mean",
    }
    if "route_length_std" in complete.columns:
        aggregations["route_length_std"] = "mean"
    if "gap_variation_pct" in complete.columns:
        aggregations["gap_variation_pct"] = "mean"
    grouped = complete.groupby(group_cols, dropna=False).agg(aggregations).reset_index()
    grouped = grouped.rename(
        columns={
            "best_len": "avg_best_len",
            "optimality_gap_pct": "avg_gap_pct",
            "convergence_time_s": "avg_convergence_time_s",
            "objective_evals_to_convergence": "avg_evals_to_convergence",
            "objective_evals_total": "avg_evals_total",
            "run_time_s": "avg_run_time_s",
            "stopped_early": "early_stop_rate",
            "optimal_match": "exact_match_rate",
        }
    )
    return grouped


def _overall_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = _aggregate_frame(rows, ["model"])
    if frame.empty:
        return []
    records = frame.to_dict(orient="records")
    for record in records:
        record["runs"] = int(sum(1 for row in rows if row.get("model") == record["model"] and row.get("status") == "complete"))
    return records


def _city_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = _aggregate_frame(rows, ["city_count", "model"])
    if frame.empty:
        return []
    return frame.sort_values(["city_count", "model"]).to_dict(orient="records")


def _scenario_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    complete = frame[(frame["status"] == "complete") & frame["best_len"].notna()].copy()
    if complete.empty:
        return []
    scenario_cols = ["city_count", "instance_seed"]
    if "variant" in complete.columns:
        scenario_cols.append("variant")
    grouped = complete.groupby(scenario_cols, dropna=False)
    records: list[dict[str, Any]] = []
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        data = dict(zip(scenario_cols, key, strict=False))
        wins = group.sort_values(["optimality_gap_pct", "run_time_s"], ascending=[True, True]).iloc[0]
        data.update(
            {
                "winner": wins["model"],
                "winner_gap_pct": _safe_float(wins.get("optimality_gap_pct")),
                "winner_runtime_s": _safe_float(wins.get("run_time_s")),
                "models_compared": int(group["model"].nunique()),
            }
        )
        records.append(data)
    return records


def _insight_lines(report: dict[str, Any]) -> list[str]:
    overall = report.get("summary_by_model") or []
    if not overall:
        return ["No completed runs were produced for this report."]
    gap_rank = sorted(overall, key=lambda item: float(item.get("avg_gap_pct", math.inf)))
    time_rank = sorted(overall, key=lambda item: float(item.get("avg_convergence_time_s", math.inf)))
    runtime_rank = sorted(overall, key=lambda item: float(item.get("avg_run_time_s", math.inf)))
    exact_rank = sorted(overall, key=lambda item: float(item.get("exact_match_rate", -math.inf)), reverse=True)

    lines = [
        f"Best average route quality: {gap_rank[0]['model']} with {float(gap_rank[0]['avg_gap_pct']):.3f}% average gap.",
        f"Fastest average convergence: {time_rank[0]['model']} with {float(time_rank[0]['avg_convergence_time_s']):.4f}s.",
        f"Lowest average runtime: {runtime_rank[0]['model']} with {float(runtime_rank[0]['avg_run_time_s']):.4f}s.",
        f"Highest exact-match rate: {exact_rank[0]['model']} at {float(exact_rank[0]['exact_match_rate']) * 100.0:.1f}%.",
    ]
    if str(report.get("variant")) == "dynamic":
        lines.append("Dynamic mode is summarized using the initial mask baseline plus iteration-level adaptation statistics.")
    if str(report.get("variant")) == "noisy":
        lines.append("Noisy mode uses a fixed blocked-edge mask per instance, so comparisons stay stable across algorithms.")
    return lines


def build_mega_report(
    *,
    variant: str,
    selected_models: list[str],
    city_sizes: list[int],
    seeds: list[int],
    model_config_values: dict[str, list[int]],
    iterations: int,
    exact_timeout: int,
    clustered: bool,
    aco_settings: dict[str, Any],
    abc_settings: dict[str, Any],
    ga_settings: dict[str, Any],
    pso_settings: dict[str, Any],
    blocked_fraction: float = 0.05,
    blocked_count: int | None = None,
    penalty: float = 1e6,
    auto_relax: bool = False,
    target_gap_pct: float = 0.0,
    max_workers: int = 4,
    use_parallel: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    variant = str(variant)
    selected_models = [str(label) for label in selected_models]
    city_sizes = [int(value) for value in city_sizes]
    seeds = [int(value) for value in seeds]
    report_started = time.perf_counter()
    instances = _build_instances(city_sizes, seeds, clustered)
    tasks: list[dict[str, Any]] = []

    for instance_index, instance in enumerate(instances):
        coords = instance["coords"]
        exact = _exact_baseline_for_variant(
            coords,
            variant=variant,
            exact_timeout=int(exact_timeout),
            blocked_fraction=blocked_fraction,
            blocked_count=blocked_count,
            penalty=penalty,
            seed=instance["seed"],
            auto_relax=auto_relax,
        )
        exact_length = exact.get("length")
        exact_status = exact.get("status")
        exact_is_optimal = bool(exact.get("is_optimal", False))
        for model_label in selected_models:
            for config_value in _resolve_config_values(model_label, model_config_values):
                run_seed = int(instance["seed"]) * 100_000 + int(config_value) * 97 + sum(ord(ch) for ch in model_label)
                tasks.append(
                    {
                        "variant": variant,
                        "coords": coords,
                        "instance_seed": int(instance["seed"]),
                        "city_count": int(instance["city_count"]),
                        "model_label": model_label,
                        "config_value": int(config_value),
                        "iterations": int(iterations),
                        "run_seed": int(run_seed),
                        "noise_seed": int(instance["seed"]),
                        "blocked_fraction": float(blocked_fraction),
                        "blocked_count": None if blocked_count is None else int(blocked_count),
                        "penalty": float(penalty),
                        "auto_relax": bool(auto_relax),
                        "exact_length": None if exact_length is None else float(exact_length),
                        "exact_status": exact_status,
                        "exact_is_optimal": exact_is_optimal,
                        "target_gap_pct": float(target_gap_pct),
                        "aco_settings": dict(aco_settings),
                        "abc_settings": dict(abc_settings),
                        "ga_settings": dict(ga_settings),
                        "pso_settings": dict(pso_settings),
                        "parallel_worker": bool(use_parallel),
                    }
                )

    total_tasks = len(tasks)
    rows: list[dict[str, Any]] = []
    completed = 0
    running_elapsed = 0.0
    total_worker_capacity = max(1, int(max_workers)) if use_parallel else 1

    def emit_progress(current_task: dict[str, Any] | None = None) -> None:
        if progress_callback is None:
            return
        avg_duration = running_elapsed / completed if completed > 0 else None
        remaining_tasks = max(0, total_tasks - completed)
        if avg_duration is None:
            eta_s = None
        else:
            eta_s = float(avg_duration * remaining_tasks / float(total_worker_capacity))
        payload = {
            "type": "mega_report_progress",
            "total_tasks": total_tasks,
            "completed_tasks": completed,
            "batch_progress": completed / float(total_tasks or 1),
            "eta_s": eta_s,
            "elapsed_s": float(time.perf_counter() - report_started),
            "current_task": current_task,
            "variant": variant,
        }
        progress_callback(payload)

    emit_progress(None)
    if use_parallel and total_tasks > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_task_worker, task): task for task in tasks}
            for future in as_completed(future_map):
                task = future_map[future]
                row = future.result()
                rows.append(row)
                completed += 1
                running_elapsed += float(row.get("run_time_s") or 0.0)
                emit_progress(task)
    else:
        for task in tasks:
            row = _task_worker(task)
            rows.append(row)
            completed += 1
            running_elapsed += float(row.get("run_time_s") or 0.0)
            emit_progress(task)

    total_elapsed = float(time.perf_counter() - report_started)
    report = {
        "generated_at": datetime.utcnow(),
        "variant": variant,
        "selected_models": selected_models,
        "city_sizes": city_sizes,
        "seeds": seeds,
        "iterations": int(iterations),
        "exact_timeout": int(exact_timeout),
        "clustered": bool(clustered),
        "blocked_fraction": float(blocked_fraction),
        "blocked_count": None if blocked_count is None else int(blocked_count),
        "penalty": float(penalty),
        "auto_relax": bool(auto_relax),
        "target_gap_pct": float(target_gap_pct),
        "max_workers": int(max_workers),
        "use_parallel": bool(use_parallel),
        "total_tasks": int(total_tasks),
        "completed_tasks": int(completed),
        "elapsed_s": total_elapsed,
        "rows": rows,
        "summary_by_model": _overall_summary(rows),
        "summary_by_city": _city_summary(rows),
        "summary_by_scenario": _scenario_summary(rows),
    }
    report["insights"] = _insight_lines(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    variant_slug = variant.replace(" ", "_").lower()
    json_path = RESULTS_DIR / f"mega_report_{variant_slug}_{stamp}.json"
    pdf_path = RESULTS_DIR / f"mega_report_{variant_slug}_{stamp}.pdf"
    report["json_path"] = str(json_path)
    report["pdf_path"] = str(pdf_path)
    json_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    save_mega_report_pdf(report, pdf_path)
    return report


def _chart_figures(report: dict[str, Any]) -> list[plt.Figure]:
    figures: list[plt.Figure] = []
    summary = pd.DataFrame(report.get("summary_by_model") or [])
    city = pd.DataFrame(report.get("summary_by_city") or [])

    if not summary.empty:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        axes = axes.ravel()
        axes[0].bar(summary["model"], summary["avg_gap_pct"], color="#f97316")
        axes[0].set_title("Average Gap %")
        axes[0].set_ylabel("Gap %")
        axes[1].bar(summary["model"], summary["avg_convergence_time_s"], color="#38bdf8")
        axes[1].set_title("Average Convergence Time")
        axes[1].set_ylabel("Seconds")
        axes[2].bar(summary["model"], summary["avg_run_time_s"], color="#facc15")
        axes[2].set_title("Average Runtime")
        axes[2].set_ylabel("Seconds")
        axes[3].bar(summary["model"], summary["exact_match_rate"] * 100.0, color="#34d399")
        axes[3].set_title("Exact Match Rate")
        axes[3].set_ylabel("Percent")
        for ax in axes:
            ax.tick_params(axis="x", rotation=20)
            ax.grid(axis="y", alpha=0.25)
        figures.append(fig)

    if not city.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for model in sorted(city["model"].unique()):
            subset = city[city["model"] == model].sort_values("city_count")
            axes[0].plot(subset["city_count"], subset["avg_gap_pct"], marker="o", label=model)
            axes[1].plot(subset["city_count"], subset["avg_run_time_s"], marker="o", label=model)
        axes[0].set_title("Gap by City Size")
        axes[0].set_xlabel("Cities")
        axes[0].set_ylabel("Gap %")
        axes[1].set_title("Runtime by City Size")
        axes[1].set_xlabel("Cities")
        axes[1].set_ylabel("Seconds")
        axes[0].grid(alpha=0.25)
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="best")
        figures.append(fig)

    row_frame = pd.DataFrame(report.get("rows") or [])
    if not row_frame.empty:
        complete = row_frame[(row_frame["status"] == "complete") & row_frame["best_len"].notna()].copy()
        if not complete.empty:
            fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
            models = sorted(complete["model"].unique())
            data = [complete[complete["model"] == model]["optimality_gap_pct"].dropna().astype(float).tolist() for model in models]
            ax.boxplot(data, labels=models, showmeans=True)
            ax.set_title("Gap Distribution by Model")
            ax.set_ylabel("Gap %")
            ax.grid(axis="y", alpha=0.25)
            figures.append(fig)
    return figures


def save_mega_report_pdf(report: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figures = _chart_figures(report)
    with PdfPages(output_path) as pdf:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        title = f"Mega TSP Report - {str(report.get('variant', 'standard')).title()}"
        lines = [
            title,
            "",
            f"Generated: {report.get('generated_at')}",
            f"Models: {', '.join(report.get('selected_models') or [])}",
            f"City sizes: {', '.join(map(str, report.get('city_sizes') or []))}",
            f"Seeds: {', '.join(map(str, report.get('seeds') or []))}",
            f"Completed tasks: {report.get('completed_tasks')} / {report.get('total_tasks')}",
            f"Elapsed: {float(report.get('elapsed_s') or 0.0):.1f}s",
            "",
            "Key findings:",
        ]
        lines.extend(f"- {item}" for item in (report.get("insights") or []))
        ax.text(0.04, 0.96, "\n".join(lines), va="top", ha="left", fontsize=12, family="DejaVu Sans")
        pdf.savefig(fig)
        plt.close(fig)

        for figure in figures:
            pdf.savefig(figure)
            plt.close(figure)
    return output_path


def report_download_bytes(report: dict[str, Any]) -> bytes:
    pdf_path = Path(str(report.get("pdf_path") or ""))
    if pdf_path.exists():
        return pdf_path.read_bytes()
    temp_path = RESULTS_DIR / "_temp_report.pdf"
    save_mega_report_pdf(report, temp_path)
    return temp_path.read_bytes()


def load_report(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data

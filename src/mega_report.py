from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import json
import time
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
from .comparison import AVAILABLE_ALGORITHMS, run_algorithm_on_instance
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


def _run_exact(coords: np.ndarray, timeout_s: int, blocked_mask: np.ndarray | None) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = solve_tsp_exact(coords, max_seconds=timeout_s, blocked_mask=blocked_mask)
        result["runtime_s"] = float(time.perf_counter() - started)
        return result
    except Exception as exc:
        elapsed = float(time.perf_counter() - started)
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
            exact = _run_exact(coords, exact_timeout, blocked_mask)
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


def _task_worker(task: dict[str, Any]) -> dict[str, Any]:
    dynamic_env.set_params(
        mode=str(task["tsp_mode"]),
        blocked_fraction=float(task["blocked_fraction"]),
        blocked_count=None if task.get("blocked_count") is None else int(task["blocked_count"]),
        penalty=float(task["penalty"]),
        seed=task.get("mode_seed"),
        auto_relax=bool(task.get("auto_relax", False)),
    )
    row = run_algorithm_on_instance(
        str(task["model_label"]),
        np.asarray(task["coords"], dtype=float),
        int(task["config_value"]),
        iterations=int(task["iterations"]),
        seed=None if task.get("city_seed") is None else int(task["city_seed"]),
        exact_length=task.get("exact_length"),
        exact_status=task.get("exact_status"),
        aco_settings=dict(task.get("aco_settings", {})),
        pso_settings=dict(task.get("pso_settings", {})),
        abc_settings=dict(task.get("abc_settings", {})),
        ga_settings=dict(task.get("ga_settings", {})),
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


def _tasks_for_scenario(
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
    blocked_fraction: float,
    blocked_count: int | None,
    penalty: float,
    auto_relax: bool,
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
                    "blocked_fraction": float(blocked_fraction),
                    "blocked_count": blocked_count,
                    "penalty": float(penalty),
                    "auto_relax": bool(auto_relax),
                }
            )
    return tasks


def _task_batch_for_scenarios(
    scenarios: Iterable[MegaReportScenario],
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
    blocked_fraction: float,
    blocked_count: int | None,
    penalty: float,
    auto_relax: bool,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for scenario in scenarios:
        tasks.extend(
            _tasks_for_scenario(
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
                blocked_fraction=blocked_fraction,
                blocked_count=blocked_count,
                penalty=penalty,
                auto_relax=auto_relax,
            )
        )
    return tasks


def _row_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    for column in ["best_len", "optimality_gap_pct", "convergence_time_s", "objective_evals_to_convergence", "objective_evals_total", "run_time_s"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def build_mega_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = _row_df(rows)
    if frame.empty:
        return []
    complete = frame[(frame["status"] == "complete") & frame["best_len"].notna()].copy()
    if complete.empty:
        return []
    summary: list[dict[str, Any]] = []
    for model, group in complete.groupby("model"):
        optimal = group.get("optimal_match", pd.Series(dtype=bool)).fillna(False)
        stopped = group.get("stopped_early", pd.Series(dtype=bool)).fillna(False)
        summary.append(
            {
                "model": model,
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
    return summary


def _city_summary(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = _row_df(rows)
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


def _scenario_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = _row_df(rows)
    if frame.empty:
        return []
    complete = frame[(frame["status"] == "complete") & frame["best_len"].notna()].copy()
    if complete.empty:
        return []
    summary: list[dict[str, Any]] = []
    for scenario_id, group in complete.groupby("scenario_id"):
        best = group.sort_values(["optimality_gap_pct", "run_time_s"], ascending=[True, True]).iloc[0]
        summary.append(
            {
                "scenario_id": scenario_id,
                "city_count": int(best["city_count"]),
                "winner": best["model"],
                "winner_gap_pct": float(best.get("optimality_gap_pct") or 0.0),
                "winner_runtime_s": float(best.get("run_time_s") or 0.0),
                "models_compared": int(group["model"].nunique()),
            }
        )
    return summary


def _build_insights(report: dict[str, Any]) -> list[str]:
    summary = report.get("summary") or []
    if not summary:
        return ["No complete runs were produced."]
    gap_best = min(summary, key=lambda row: row["avg_gap_pct"])
    time_best = min(summary, key=lambda row: row["avg_run_time_s"])
    eval_best = min(summary, key=lambda row: row["avg_evals_to_convergence"])
    match_best = max(summary, key=lambda row: row["exact_match_rate"])
    lines = [
        f"Best average route quality: {gap_best['model']} with an average gap of {gap_best['avg_gap_pct']:.3f}%.",
        f"Fastest end-to-end runtime: {time_best['model']} with {time_best['avg_run_time_s']:.4f}s on average.",
        f"Lowest effort to convergence: {eval_best['model']} with {eval_best['avg_evals_to_convergence']:.1f} evaluations on average.",
        f"Highest exact-match rate: {match_best['model']} at {match_best['exact_match_rate'] * 100.0:.1f}%.",
    ]
    if str(report.get("variant")) == "dynamic":
        lines.append("Dynamic mode is summarized from the initial mask baseline plus the run-time adaptation metrics.")
    if str(report.get("variant")) == "noisy":
        lines.append("Noisy mode uses a fixed blocked-edge mask per instance, so comparisons stay stable across algorithms.")
    return lines


def _make_pdf(report: dict[str, Any], pdf_path: Path) -> None:
    summary = pd.DataFrame(report.get("summary", []))
    city_gap_rows = report.get("city_gap_rows") or report.get("rows", [])
    city_summary = _city_summary(city_gap_rows)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis("off")
        fig.text(0.06, 0.95, f"Mega TSP Report: {str(report.get('variant', 'standard')).title()}", fontsize=20, fontweight="bold")
        fig.text(0.06, 0.91, f"Created: {report.get('created_at')}", fontsize=10)
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
            axes[1].bar(summary_sorted["model"], summary_sorted["avg_run_time_s"], color="#38bdf8")
            axes[1].set_title("Average runtime (s)")
            axes[2].bar(summary_sorted["model"], summary_sorted["avg_evals_to_convergence"], color="#a78bfa")
            axes[2].set_title("Average evals to convergence")
            axes[3].bar(summary_sorted["model"], summary_sorted["exact_match_rate"] * 100.0, color="#34d399")
            axes[3].set_title("Exact-match rate (%)")
            for ax in axes:
                ax.tick_params(axis="x", rotation=20)
                ax.grid(axis="y", alpha=0.25)
            fig.suptitle("Model Summary", fontsize=18, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if not city_summary.empty:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            for model in sorted(city_summary["model"].unique()):
                subset = city_summary[city_summary["model"] == model].sort_values("city_count")
                ax.plot(subset["city_count"], subset["avg_gap_pct"], marker="o", label=model)
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
    city_gap_detail_mode: str = "summary",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    report_id = _make_report_id()
    report_dir = REPORT_ROOT / report_id
    report_dir.mkdir(parents=True, exist_ok=True)

    selected_models = [str(model) for model in selected_models] or [spec.label for spec in AVAILABLE_ALGORITHMS if spec.runnable]
    city_sizes = [int(value) for value in city_sizes]
    detail_mode = str(city_gap_detail_mode).strip().lower()
    detailed_city_gap = detail_mode in {"detailed", "detail", "full", "dense", "all"}
    scenarios = _build_scenarios(
        city_sizes,
        instances_per_size=int(instances_per_size),
        base_seed=base_seed,
        clustered=clustered,
        tsp_mode=tsp_mode,
        exact_timeout=int(exact_timeout),
        blocked_fraction=float(blocked_fraction),
        blocked_count=blocked_count,
        penalty=float(penalty),
        auto_relax=bool(auto_relax),
    )
    detailed_city_sizes = list(range(30, 41)) if detailed_city_gap else []
    detailed_scenarios = _build_scenarios(
        detailed_city_sizes,
        instances_per_size=int(instances_per_size),
        base_seed=base_seed,
        clustered=clustered,
        tsp_mode=tsp_mode,
        exact_timeout=int(exact_timeout),
        blocked_fraction=float(blocked_fraction),
        blocked_count=blocked_count,
        penalty=float(penalty),
        auto_relax=bool(auto_relax),
    ) if detailed_city_gap else []
    tasks = _task_batch_for_scenarios(
        scenarios,
        selected_models=selected_models,
        model_config_values=model_config_values,
        iterations=iterations,
        target_gap_pct=target_gap_pct,
        aco_settings=aco_settings,
        pso_settings=pso_settings,
        abc_settings=abc_settings,
        ga_settings=ga_settings,
        tsp_mode=tsp_mode,
        blocked_fraction=blocked_fraction,
        blocked_count=blocked_count,
        penalty=penalty,
        auto_relax=auto_relax,
    )
    detailed_tasks = _task_batch_for_scenarios(
        detailed_scenarios,
        selected_models=selected_models,
        model_config_values=model_config_values,
        iterations=iterations,
        target_gap_pct=target_gap_pct,
        aco_settings=aco_settings,
        pso_settings=pso_settings,
        abc_settings=abc_settings,
        ga_settings=ga_settings,
        tsp_mode=tsp_mode,
        blocked_fraction=blocked_fraction,
        blocked_count=blocked_count,
        penalty=penalty,
        auto_relax=auto_relax,
    ) if detailed_city_gap else []

    rows: list[dict[str, Any]] = []
    city_gap_rows: list[dict[str, Any]] = []
    total_work = max(1, len(scenarios) + len(tasks) + len(detailed_scenarios) + len(detailed_tasks))
    completed = 0
    started = time.perf_counter()

    def emit(event: dict[str, Any]) -> None:
        if progress_callback is None:
            return
        elapsed = max(1e-9, time.perf_counter() - started)
        eta_s = None
        if completed > 0:
            eta_s = elapsed / completed * max(0, total_work - completed)
        progress_callback(
            {
                "type": event.get("type", "progress"),
                "completed_work": completed,
                "total_work": total_work,
                "progress": completed / float(total_work),
                "eta_s": eta_s,
                **event,
            }
        )

    emit({"type": "prep_started", "scenario_count": len(scenarios), "task_count": len(tasks)})
    for index, scenario in enumerate(scenarios, start=1):
        completed += 1
        emit(
            {
                "type": "scenario_prepared",
                "scenario_index": index,
                "scenario_total": len(scenarios),
                "scenario_id": scenario.scenario_id,
                "city_count": scenario.city_count,
            }
        )

    if detailed_city_gap:
        emit({"type": "gap_prep_started", "scenario_count": len(detailed_scenarios), "task_count": len(detailed_tasks), "city_gap_detail_mode": detail_mode})
        for index, scenario in enumerate(detailed_scenarios, start=1):
            completed += 1
            emit(
                {
                    "type": "gap_scenario_prepared",
                    "scenario_index": index,
                    "scenario_total": len(detailed_scenarios),
                    "scenario_id": scenario.scenario_id,
                    "city_count": scenario.city_count,
                }
            )

    if parallel_workers <= 1 or len(tasks) <= 1:
        for index, task in enumerate(tasks, start=1):
            rows.append(_task_worker(task))
            completed += 1
            emit({"type": "task_completed", "task_index": index, "task_total": len(tasks), "model": task["model_label"], "scenario_id": task["scenario_id"]})
    else:
        with ProcessPoolExecutor(max_workers=int(parallel_workers)) as executor:
            future_map = {executor.submit(_task_worker, task): task for task in tasks}
            for index, future in enumerate(as_completed(future_map), start=1):
                task = future_map[future]
                rows.append(future.result())
                completed += 1
                emit({"type": "task_completed", "task_index": index, "task_total": len(tasks), "model": task["model_label"], "scenario_id": task["scenario_id"]})

    if detailed_city_gap:
        if parallel_workers <= 1 or len(detailed_tasks) <= 1:
            for index, task in enumerate(detailed_tasks, start=1):
                city_gap_rows.append(_task_worker(task))
                completed += 1
                emit({"type": "gap_task_completed", "task_index": index, "task_total": len(detailed_tasks), "model": task["model_label"], "scenario_id": task["scenario_id"]})
        else:
            with ProcessPoolExecutor(max_workers=int(parallel_workers)) as executor:
                future_map = {executor.submit(_task_worker, task): task for task in detailed_tasks}
                for index, future in enumerate(as_completed(future_map), start=1):
                    task = future_map[future]
                    city_gap_rows.append(future.result())
                    completed += 1
                    emit({"type": "gap_task_completed", "task_index": index, "task_total": len(detailed_tasks), "model": task["model_label"], "scenario_id": task["scenario_id"]})

    report = {
        "report_id": report_id,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "variant": tsp_mode,
        "settings": {
            "tsp_mode": tsp_mode,
            "city_sizes": city_sizes,
            "city_gap_detail_mode": detail_mode,
            "city_gap_detail_sizes": detailed_city_sizes,
            "instances_per_size": int(instances_per_size),
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
        "city_gap_rows": city_gap_rows,
    }
    report["summary"] = build_mega_summary(rows)
    report["scenario_summary"] = _scenario_summary(rows)
    report["insights"] = _build_insights(report)

    json_path = report_dir / "report.json"
    pdf_path = report_dir / "report.pdf"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    _make_pdf(report, pdf_path)
    report["files"] = {"json": str(json_path), "pdf": str(pdf_path)}
    report["elapsed_s"] = float(time.perf_counter() - started)
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
    return _row_df(report.get("rows", []))


def city_summary_frame(report: dict[str, Any], *, use_detailed: bool = False) -> pd.DataFrame:
    if use_detailed and report.get("city_gap_rows"):
        return _city_summary(report.get("city_gap_rows", []))
    return _city_summary(report.get("rows", []))

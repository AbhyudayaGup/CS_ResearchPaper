from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Sequence
import time

import numpy as np

from .aco import AntColony
from .exact_solver import solve_tsp_exact
from .pso import ParticleSwarm
from .utils import generate_cities


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    label: str
    config_label: str
    runnable: bool
    description: str


AVAILABLE_ALGORITHMS: List[AlgorithmSpec] = [
    AlgorithmSpec(
        key="aco",
        label="ACO",
        config_label="ants",
        runnable=True,
        description="Ant Colony Optimization with optional 2-opt local search.",
    ),
    AlgorithmSpec(
        key="pso",
        label="PSO",
        config_label="particles",
        runnable=True,
        description="Particle Swarm Optimization using random-key permutation decoding.",
    ),
    AlgorithmSpec(
        key="bee",
        label="Bee Colony",
        config_label="bees",
        runnable=False,
        description="Coming soon.",
    ),
]


def spec_by_label(label: str) -> AlgorithmSpec:
    for spec in AVAILABLE_ALGORITHMS:
        if spec.label == label:
            return spec
    raise KeyError(f"Unknown algorithm label: {label}")


def default_algorithm_labels() -> List[str]:
    return [spec.label for spec in AVAILABLE_ALGORITHMS if spec.runnable]


def parse_int_list(raw: str, fallback: Sequence[int]) -> List[int]:
    values: List[int] = []
    for part in (raw or "").replace(";", ",").split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            values.append(int(stripped))
        except ValueError:
            continue
    if not values:
        values = list(fallback)
    cleaned = []
    for value in values:
        if value not in cleaned:
            cleaned.append(int(value))
    return cleaned


def build_city_instances(
    city_sizes: Sequence[int],
    seed: int | None,
    clustered: bool,
) -> List[Dict[str, Any]]:
    base_seed = 0 if seed is None else int(seed)
    instances: List[Dict[str, Any]] = []
    for index, city_count in enumerate(city_sizes):
        instance_seed = None if seed is None else base_seed + index * 101
        coords = generate_cities(int(city_count), seed=instance_seed, clustered=clustered)
        instances.append(
            {
                "city_count": int(city_count),
                "seed": instance_seed,
                "coords": coords,
            }
        )
    return instances


def run_exact_baseline(coords: np.ndarray, max_seconds: int) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        result = solve_tsp_exact(coords, max_seconds=max_seconds)
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


def _extract_run_result(run_result: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    best_tour = best_len = history = convergence_iteration = best_found_iter = stats = None
    if isinstance(run_result, tuple):
        if len(run_result) == 3:
            best_tour, best_len, history = run_result
        elif len(run_result) == 4:
            best_tour, best_len, history, convergence_iteration = run_result
        elif len(run_result) >= 5:
            best_tour, best_len, history, convergence_iteration, best_found_iter = run_result[:5]
        if len(run_result) >= 6:
            stats = run_result[5]
    return best_tour, best_len, history, convergence_iteration, best_found_iter, stats


def _fallback_stats(model_key: str, history, best_len, best_found_iter, convergence_iteration, elapsed_s, population_size):
    history_len = len(history) if history else 0
    if best_found_iter is not None:
        best_iter = int(best_found_iter)
    elif convergence_iteration is not None:
        best_iter = int(convergence_iteration)
    else:
        best_iter = max(0, history_len - 1)

    if history_len > 0:
        fraction = float(best_iter + 1) / float(history_len)
    else:
        fraction = 1.0

    objective_evals_total = int(max(1, population_size * (history_len + (1 if model_key == "pso" else 0))))
    objective_evals_to_convergence = int(max(1, population_size * (best_iter + 1)))

    return {
        "run_time_s": float(elapsed_s),
        "convergence_time_s": float(max(0.0, elapsed_s * min(1.0, max(0.0, fraction)))),
        "objective_evals_total": objective_evals_total,
        "objective_evals_to_convergence": objective_evals_to_convergence,
        "best_found_iteration": int(best_iter),
        "iterations_executed": int(history_len),
        "stopped_early": bool(history_len > 0 and best_found_iter is not None and best_found_iter < history_len - 1),
    }


def _run_aco(coords: np.ndarray, config_value: int, *, iterations: int, seed: int | None, two_opt: bool, alpha: float, beta: float, rho: float, Q: float, target_length: float | None = None, target_gap_pct: float = 0.0, progress_callback: Callable[[Dict[str, Any]], None] | None = None):
    model = AntColony(
        coords,
        n_ants=int(config_value),
        n_iterations=int(iterations),
        alpha=float(alpha),
        beta=float(beta),
        rho=float(rho),
        Q=float(Q),
        apply_two_opt=bool(two_opt),
        seed=seed,
    )
    started = time.perf_counter()
    def _callback(**payload: Any) -> None:
        if progress_callback is not None:
            progress_callback(payload)

    run_result = model.run(
        callback=_callback if progress_callback is not None else None,
        return_stats=True,
        target_length=target_length,
        target_gap_pct=target_gap_pct,
    )
    elapsed_s = time.perf_counter() - started
    best_tour, best_len, history, convergence_iteration, best_found_iter, stats = _extract_run_result(run_result)
    if stats is None:
        stats = _fallback_stats("aco", history, best_len, best_found_iter, convergence_iteration, elapsed_s, int(config_value))
    return best_tour, best_len, history, convergence_iteration, best_found_iter, stats


def _run_pso(coords: np.ndarray, config_value: int, *, iterations: int, seed: int | None, two_opt: bool, w: float, c1: float, c2: float, target_length: float | None = None, target_gap_pct: float = 0.0, progress_callback: Callable[[Dict[str, Any]], None] | None = None):
    model = ParticleSwarm(
        coords,
        n_particles=int(config_value),
        n_iterations=int(iterations),
        w=float(w),
        c1=float(c1),
        c2=float(c2),
        apply_two_opt=bool(two_opt),
        seed=seed,
    )
    started = time.perf_counter()
    def _callback(**payload: Any) -> None:
        if progress_callback is not None:
            progress_callback(payload)

    run_result = model.run(
        callback=_callback if progress_callback is not None else None,
        return_stats=True,
        target_length=target_length,
        target_gap_pct=target_gap_pct,
    )
    elapsed_s = time.perf_counter() - started
    best_tour, best_len, history, convergence_iteration, best_found_iter, stats = _extract_run_result(run_result)
    if stats is None:
        stats = _fallback_stats("pso", history, best_len, best_found_iter, convergence_iteration, elapsed_s, int(config_value))
    return best_tour, best_len, history, convergence_iteration, best_found_iter, stats


def run_algorithm_on_instance(
    algorithm_label: str,
    coords: np.ndarray,
    config_value: int,
    *,
    iterations: int,
    seed: int | None,
    exact_length: float | None,
    exact_status: str | None,
    aco_settings: Dict[str, Any],
    pso_settings: Dict[str, Any],
    target_gap_pct: float = 0.0,
    progress_callback: Callable[[Dict[str, Any]], None] | None = None,
) -> Dict[str, Any]:
    spec = spec_by_label(algorithm_label)
    base: Dict[str, Any] = {
        "model": spec.label,
        "model_key": spec.key,
        "config_value": int(config_value),
        "config_label": spec.config_label,
        "iterations": int(iterations),
        "status": "complete",
        "reason": None,
        "best_len": None,
        "exact_length": None if exact_length is None else float(exact_length),
        "exact_status": exact_status,
        "optimal_match": None,
        "optimality_gap_pct": None,
        "convergence_iteration": None,
        "best_found_iteration": None,
        "convergence_time_s": None,
        "objective_evals_to_convergence": None,
        "objective_evals_total": None,
        "run_time_s": None,
        "matched_exact": None,
        "iterations_executed": None,
        "stopped_early": None,
    }

    if not spec.runnable:
        base.update(
            {
                "status": "unavailable",
                "reason": "This model is listed in the dashboard but is not implemented yet.",
            }
        )
        return base

    target_length = exact_length if exact_length is not None and exact_status == "OPTIMAL" else None

    if spec.key == "aco":
        _, best_len, _, convergence_iteration, best_found_iter, stats = _run_aco(
            coords,
            config_value,
            iterations=iterations,
            seed=seed,
            two_opt=aco_settings.get("two_opt", True),
            alpha=aco_settings.get("alpha", 1.0),
            beta=aco_settings.get("beta", 5.0),
            rho=aco_settings.get("rho", 0.5),
            Q=aco_settings.get("Q", 100.0),
            target_length=target_length,
            target_gap_pct=target_gap_pct,
            progress_callback=progress_callback,
        )
    elif spec.key == "pso":
        _, best_len, _, convergence_iteration, best_found_iter, stats = _run_pso(
            coords,
            config_value,
            iterations=iterations,
            seed=seed,
            two_opt=pso_settings.get("two_opt", True),
            w=pso_settings.get("w", 0.5),
            c1=pso_settings.get("c1", 1.5),
            c2=pso_settings.get("c2", 1.5),
            target_length=target_length,
            target_gap_pct=target_gap_pct,
            progress_callback=progress_callback,
        )
    else:
        base.update(
            {
                "status": "unavailable",
                "reason": "No runner has been wired for this algorithm yet.",
            }
        )
        return base

    if best_len is None:
        base.update(
            {
                "status": "error",
                "reason": "The optimizer did not return a valid tour length.",
            }
        )
        return base

    optimal_match = None
    gap_pct = None
    if exact_length is not None and exact_length > 0:
        gap_pct = ((float(best_len) - float(exact_length)) / float(exact_length)) * 100.0
        optimal_match = abs(float(best_len) - float(exact_length)) <= max(1e-9, float(exact_length) * 1e-6)

    base.update(
        {
            "best_len": float(best_len),
            "optimality_gap_pct": gap_pct,
            "optimal_match": optimal_match,
            "convergence_iteration": int(convergence_iteration) if convergence_iteration is not None else None,
            "best_found_iteration": int(best_found_iter) if best_found_iter is not None else None,
            "convergence_time_s": float(stats["convergence_time_s"]),
            "objective_evals_to_convergence": int(stats["objective_evals_to_convergence"]),
            "objective_evals_total": int(stats["objective_evals_total"]),
            "run_time_s": float(stats["run_time_s"]),
            "matched_exact": optimal_match,
            "iterations_executed": int(stats.get("iterations_executed", iterations)),
            "stopped_early": bool(stats.get("stopped_early", False)),
        }
    )
    return base


def run_comparison_batch(
    selected_models: Sequence[str],
    city_sizes: Sequence[int],
    model_config_values: Dict[str, Sequence[int]],
    *,
    iterations: int,
    base_seed: int | None,
    clustered: bool,
    exact_timeout: int,
    aco_settings: Dict[str, Any],
    pso_settings: Dict[str, Any],
    target_gap_pct: float = 0.0,
    progress_callback: Callable[[Dict[str, Any]], None] | None = None,
) -> Dict[str, Any]:
    instances = build_city_instances(city_sizes, base_seed, clustered)
    rows: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []
    for instance_index, instance in enumerate(instances):
        for model_label in selected_models:
            for config_value in model_config_values.get(model_label, [10, 20, 40]):
                tasks.append(
                    {
                        "instance_index": instance_index,
                        "instance": instance,
                        "model_label": model_label,
                        "config_value": int(config_value),
                    }
                )

    total_tasks = len(tasks)
    completed_tasks = 0

    def emit_progress(event: Dict[str, Any]) -> None:
        if progress_callback is not None:
            progress_callback({
                "batch_total": total_tasks,
                "batch_completed": completed_tasks,
                **event,
            })

    for task_index, task in enumerate(tasks):
        instance = task["instance"]
        coords = instance["coords"]
        exact = run_exact_baseline(coords, exact_timeout)
        exact_length = exact.get("length")
        exact_status = exact.get("status")

        def run_progress_callback(payload: Dict[str, Any]) -> None:
            if progress_callback is None:
                return
            iterations_total = max(1, int(iterations))
            iteration = int(payload.get("iteration", 0))
            current_progress = min(1.0, float(iteration + 1) / float(iterations_total))
            batch_progress = (completed_tasks + current_progress) / float(total_tasks or 1)
            emit_progress(
                {
                    "type": "iteration",
                    "task_index": task_index,
                    "task_total": total_tasks,
                    "model": task["model_label"],
                    "city_count": int(instance["city_count"]),
                    "config_value": int(task["config_value"]),
                    "config_label": model_config_values.get(task["model_label"], [])[0] if model_config_values.get(task["model_label"]) else None,
                    "iteration": iteration + 1,
                    "iterations_total": iterations_total,
                    "current_progress": current_progress,
                    "batch_progress": batch_progress,
                    "best_len": payload.get("best_len"),
                    "convergence_iteration": payload.get("convergence_iteration"),
                }
            )

        run_seed = None if base_seed is None else int(base_seed) + task["instance_index"] * 1000 + int(task["config_value"])
        emit_progress(
            {
                "type": "run_started",
                "task_index": task_index,
                "task_total": total_tasks,
                "model": task["model_label"],
                "city_count": int(instance["city_count"]),
                "config_value": int(task["config_value"]),
                "exact_status": exact_status,
                "exact_length": None if exact_length is None else float(exact_length),
                "batch_progress": completed_tasks / float(total_tasks or 1),
                "current_progress": 0.0,
            }
        )
        row = run_algorithm_on_instance(
            task["model_label"],
            coords,
            int(task["config_value"]),
            iterations=iterations,
            seed=run_seed,
            exact_length=exact_length,
            exact_status=exact_status,
            aco_settings=aco_settings,
            pso_settings=pso_settings,
            target_gap_pct=target_gap_pct,
            progress_callback=run_progress_callback,
        )
        row.update(
            {
                "city_count": int(instance["city_count"]),
                "city_seed": instance["seed"],
                "exact_status": exact_status,
                "exact_length": None if exact_length is None else float(exact_length),
                "exact_runtime_s": float(exact.get("runtime_s", exact.get("solve_time_s", 0.0))),
                "exact_is_optimal": bool(exact.get("is_optimal", False)),
                "exact_method": exact.get("method"),
            }
        )
        rows.append(row)
        completed_tasks += 1
        emit_progress(
            {
                "type": "run_completed",
                "task_index": task_index,
                "task_total": total_tasks,
                "model": task["model_label"],
                "city_count": int(instance["city_count"]),
                "config_value": int(task["config_value"]),
                "status": row.get("status"),
                "stopped_early": row.get("stopped_early"),
                "batch_progress": completed_tasks / float(total_tasks or 1),
                "current_progress": 1.0,
                "exact_status": exact_status,
            }
        )

    emit_progress(
        {
            "type": "batch_completed",
            "task_total": total_tasks,
            "batch_progress": 1.0,
            "current_progress": 1.0,
        }
    )

    return {
        "instances": instances,
        "rows": rows,
    }


def build_model_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    complete_rows = [row for row in rows if row.get("status") == "complete" and row.get("best_len") is not None]
    summaries: List[Dict[str, Any]] = []
    for model_label in {row["model"] for row in complete_rows}:
        model_rows = [row for row in complete_rows if row["model"] == model_label]
        if not model_rows:
            continue
        avg_gap = sum((row["optimality_gap_pct"] or 0.0) for row in model_rows) / len(model_rows)
        avg_time = sum(float(row["convergence_time_s"] or 0.0) for row in model_rows) / len(model_rows)
        avg_evals = sum(float(row["objective_evals_to_convergence"] or 0.0) for row in model_rows) / len(model_rows)
        optimal_hits = sum(1 for row in model_rows if row.get("optimal_match"))
        summaries.append(
            {
                "model": model_label,
                "avg_gap_pct": avg_gap,
                "avg_convergence_time_s": avg_time,
                "avg_evals_to_convergence": avg_evals,
                "optimal_hits": optimal_hits,
                "runs": len(model_rows),
            }
        )
    return summaries


def build_insights(rows: Sequence[Dict[str, Any]]) -> List[str]:
    complete_rows = [row for row in rows if row.get("status") == "complete" and row.get("best_len") is not None]
    if not complete_rows:
        return ["Run at least one configured model to generate comparative insights."]

    summaries = build_model_summary(rows)
    if not summaries:
        return ["No comparable model summaries were produced."]

    by_gap = min(summaries, key=lambda row: row["avg_gap_pct"])
    by_time = min(summaries, key=lambda row: row["avg_convergence_time_s"])
    by_evals = min(summaries, key=lambda row: row["avg_evals_to_convergence"])
    by_hits = max(summaries, key=lambda row: row["optimal_hits"])

    max_gap = max(summary["avg_gap_pct"] for summary in summaries)
    min_gap = min(summary["avg_gap_pct"] for summary in summaries)

    model_count = len(summaries)
    if len({summary["model"] for summary in summaries}) == 1:
        only = summaries[0]
        return [
            f"{only['model']} completed all compared runs. With only one active model selected, the dashboard is showing its internal behavior rather than a head-to-head winner.",
        ]

    insights = [
        f"Best route quality overall: {by_gap['model']} with an average optimality gap of {by_gap['avg_gap_pct']:.3f}%. That means it produced the closest routes to the exact baseline across the selected scenarios.",
        f"Fastest convergence overall: {by_time['model']} at {by_time['avg_convergence_time_s']:.4f}s on average. It is better where quick usable answers matter more than squeezing out the last bit of route quality.",
        f"Lowest compute effort to convergence: {by_evals['model']} with {by_evals['avg_evals_to_convergence']:.0f} objective evaluations on average. Fewer evaluations usually means less search work before the model settles.",
        f"Most exact matches: {by_hits['model']} hit the exact baseline {by_hits['optimal_hits']} times out of {by_hits['runs']} runs. That is the strongest signal of route quality consistency.",
    ]

    if by_gap["model"] != by_time["model"]:
        insights.append(
            f"The quality leader and speed leader are different, which is normal: the quality-focused model likely spends more work refining tours, while the speed-focused model prioritizes faster movement through the search space."
        )
    else:
        insights.append(
            f"{by_gap['model']} wins on both quality and speed here, so it is the strongest choice for this particular problem setup among the models you selected."
        )

    if max_gap - min_gap > 1.0:
        insights.append(
            "The gap spread is large enough to matter, so model choice changes route quality meaningfully rather than just cosmetically."
        )

    if len(complete_rows) >= 8:
        insights.append(
            "Because the report spans multiple city sizes and parameter settings, these patterns are more reliable than a single-run comparison and better reflect scalability.")

    return insights

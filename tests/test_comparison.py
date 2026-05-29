from __future__ import annotations

import numpy as np

from src.comparison import run_algorithm_on_instance


def test_run_algorithm_uses_exact_target_only_when_optimal(monkeypatch):
    captured = []

    def fake_run_aco(*args, **kwargs):
        captured.append(kwargs)
        return None, 10.0, [10.0], 0, 0, {
            "run_time_s": 0.1,
            "convergence_time_s": 0.05,
            "objective_evals_total": 12,
            "objective_evals_to_convergence": 4,
            "convergence_iteration": 0,
            "best_found_iteration": 0,
            "iterations_executed": 1,
            "stopped_early": False,
        }

    monkeypatch.setattr("src.comparison._run_aco", fake_run_aco)

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    base_kwargs = {
        "iterations": 10,
        "seed": 7,
        "aco_settings": {"two_opt": True, "alpha": 1.0, "beta": 5.0, "rho": 0.5, "Q": 100.0},
        "pso_settings": {"two_opt": True, "w": 0.5, "c1": 1.5, "c2": 1.5},
    }

    run_algorithm_on_instance(
        "ACO",
        coords,
        10,
        exact_length=9.5,
        exact_status="FEASIBLE",
        target_gap_pct=0.75,
        **base_kwargs,
    )
    assert captured[-1]["target_length"] is None
    assert captured[-1]["target_gap_pct"] == 0.75

    run_algorithm_on_instance(
        "ACO",
        coords,
        10,
        exact_length=9.5,
        exact_status="OPTIMAL",
        target_gap_pct=0.0,
        **base_kwargs,
    )
    assert captured[-1]["target_length"] == 9.5
    assert captured[-1]["target_gap_pct"] == 0.0


def test_run_algorithm_uses_exact_target_only_when_optimal_for_abc_and_ga(monkeypatch):
    captured = []

    def fake_run_abc(*args, **kwargs):
        captured.append(("ABC", kwargs))
        return None, 10.0, [10.0], 0, 0, {
            "run_time_s": 0.1,
            "convergence_time_s": 0.05,
            "objective_evals_total": 12,
            "objective_evals_to_convergence": 4,
            "convergence_iteration": 0,
            "best_found_iteration": 0,
            "iterations_executed": 1,
            "stopped_early": False,
        }

    def fake_run_ga(*args, **kwargs):
        captured.append(("GA", kwargs))
        return None, 10.0, [10.0], 0, 0, {
            "run_time_s": 0.1,
            "convergence_time_s": 0.05,
            "objective_evals_total": 12,
            "objective_evals_to_convergence": 4,
            "convergence_iteration": 0,
            "best_found_iteration": 0,
            "iterations_executed": 1,
            "stopped_early": False,
        }

    monkeypatch.setattr("src.comparison._run_abc", fake_run_abc)
    monkeypatch.setattr("src.comparison._run_ga", fake_run_ga)

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    base_kwargs = {
        "iterations": 10,
        "seed": 7,
        "aco_settings": {"two_opt": True, "alpha": 1.0, "beta": 5.0, "rho": 0.5, "Q": 100.0},
        "pso_settings": {"two_opt": True, "w": 0.5, "c1": 1.5, "c2": 1.5},
    }

    run_algorithm_on_instance(
        "ABC",
        coords,
        10,
        exact_length=9.5,
        exact_status="FEASIBLE",
        target_gap_pct=0.75,
        **base_kwargs,
    )
    assert captured[-1][0] == "ABC"
    assert captured[-1][1]["target_length"] is None
    assert captured[-1][1]["target_gap_pct"] == 0.75

    run_algorithm_on_instance(
        "GA",
        coords,
        10,
        exact_length=9.5,
        exact_status="OPTIMAL",
        target_gap_pct=0.0,
        **base_kwargs,
    )
    assert captured[-1][0] == "GA"
    assert captured[-1][1]["target_length"] == 9.5
    assert captured[-1][1]["target_gap_pct"] == 0.0

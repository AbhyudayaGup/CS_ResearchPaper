from __future__ import annotations

from pathlib import Path

import numpy as np

from src import mega_report as mega


def test_run_mega_report_creates_json_and_pdf(monkeypatch, tmp_path):
    monkeypatch.setattr(mega, "REPORT_ROOT", tmp_path)

    scenario = mega.MegaReportScenario(
        scenario_id="3:0",
        city_count=3,
        instance_seed=11,
        mode_seed=23,
        coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        exact_length=12.0,
        exact_status="OPTIMAL",
        exact_runtime_s=0.01,
        exact_is_optimal=True,
        exact_method="cp_sat",
    )

    monkeypatch.setattr(mega, "_build_scenarios", lambda *args, **kwargs: [scenario])

    def fake_task_worker(task: dict[str, object]) -> dict[str, object]:
        model = str(task["model_label"])
        config_value = int(task["config_value"])
        gap = float(config_value) / 10.0
        return {
            "model": model,
            "status": "complete",
            "best_len": 10.0 + gap,
            "optimality_gap_pct": gap,
            "convergence_time_s": 0.1 + gap,
            "objective_evals_to_convergence": 5 + config_value,
            "objective_evals_total": 10 + config_value,
            "run_time_s": 0.2 + gap,
            "stopped_early": False,
            "optimal_match": model == "ACO",
            "scenario_id": str(task["scenario_id"]),
            "city_count": int(task["city_count"]),
            "city_seed": task.get("city_seed"),
            "mode_seed": task.get("mode_seed"),
        }

    monkeypatch.setattr(mega, "_task_worker", fake_task_worker)

    report = mega.run_mega_report(
        tsp_mode="noisy",
        city_sizes=[3],
        instances_per_size=1,
        base_seed=7,
        clustered=False,
        iterations=10,
        exact_timeout=5,
        selected_models=["ACO", "ABC", "GA", "PSO"],
        model_config_values={"ACO": [10], "ABC": [10], "GA": [10], "PSO": [10]},
        aco_settings={"two_opt": True, "alpha": 1.0, "beta": 5.0, "rho": 0.5, "Q": 100.0},
        pso_settings={"two_opt": True, "w": 0.5, "c1": 1.5, "c2": 1.5},
        abc_settings={"two_opt": True, "limit": 10},
        ga_settings={"two_opt": True, "crossover_rate": 0.9, "mutation_rate": 0.2, "elite_fraction": 0.1, "tournament_size": 3},
        blocked_fraction=0.05,
        blocked_count=None,
        penalty=1e6,
        auto_relax=False,
        target_gap_pct=0.0,
        parallel_workers=1,
    )

    json_path = Path(report["files"]["json"])
    pdf_path = Path(report["files"]["pdf"])
    assert report["variant"] == "noisy"
    assert json_path.exists()
    assert pdf_path.exists()
    assert len(report["summary"]) == 4
    assert report["summary"][0]["runs"] == 1
    assert report["scenario_summary"][0]["winner"] in {"ACO", "ABC", "GA", "PSO"}

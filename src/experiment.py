import itertools
import time
from .utils import generate_cities
from .aco import AntColony


def sweep(params):
    keys = [k for k in params if isinstance(params[k], (list, tuple))]
    fixed = {k: v for k, v in params.items() if k not in keys}
    rows = []
    lists = [params[k] for k in keys]
    for combo in itertools.product(*lists):
        combo_dict = dict(zip(keys, combo))
        cfg = {**fixed, **combo_dict}
        for seed in range(cfg.get("n_seeds", 3)):
            coords = generate_cities(cfg.get("num_cities", 20), seed=seed)
            ac = AntColony(coords,
                           n_ants=cfg.get("num_ants", None),
                           n_iterations=cfg.get("iterations", 200),
                           alpha=cfg.get("alpha", 1.0),
                           beta=cfg.get("beta", 5.0),
                           rho=cfg.get("rho", 0.5),
                           Q=cfg.get("Q", 100.0),
                           apply_two_opt=cfg.get("apply_two_opt", True),
                           seed=seed,
                           elitist_weight=cfg.get("elitist_weight", 0.0))
            t0 = time.time()
            result = ac.run(return_stats=True)
            best_t = best_len = hist = conv_it = best_found_iter = stats = None
            if isinstance(result, tuple):
                if len(result) == 3:
                    best_t, best_len, hist = result
                elif len(result) == 4:
                    best_t, best_len, hist, conv_it = result
                elif len(result) >= 5:
                    best_t, best_len, hist, conv_it, best_found_iter = result[:5]
                if len(result) >= 6:
                    stats = result[5]
            else:
                best_t, best_len, hist = result
            t1 = time.time()
            row = {**cfg}
            row.update({
                "seed": seed,
                "best_len": float(best_len),
                "convergence_iteration": int(conv_it) if conv_it is not None else None,
                "best_found_iteration": int(best_found_iter) if best_found_iter is not None else None,
                "convergence_time_s": float(stats["convergence_time_s"]) if stats else None,
                "objective_evals_to_convergence": int(stats["objective_evals_to_convergence"]) if stats else None,
                "objective_evals_total": int(stats["objective_evals_total"]) if stats else None,
                "time_s": t1 - t0,
            })
            rows.append(row)
    return rows


if __name__ == "__main__":
    # Example sweep
    params = {
        "num_cities": 20,
        "n_seeds": 3,
        "alpha": [0.5, 1.0, 2.0],
        "beta": [2, 5],
        "rho": [0.1, 0.5],
        "iterations": 200,
        "num_ants": None,
        "Q": 100.0,
        "apply_two_opt": True,
    }
    results = sweep(params)
    print(f"Generated {len(results)} comparison rows in memory.")

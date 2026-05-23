import itertools
import time
import csv
import os
from .utils import generate_cities
from .aco import AntColony


def sweep(params, output_csv="results/experiments.csv", ensure_dir=True):
    if ensure_dir:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    keys = [k for k in params if isinstance(params[k], (list, tuple))]
    fixed = {k: v for k, v in params.items() if k not in keys}

    fieldnames = list(params.keys()) + ["seed", "best_len", "convergence_iteration", "time_s"]
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
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
                result = ac.run()
                best_t = best_len = hist = conv_it = best_found_iter = None
                if isinstance(result, tuple):
                    if len(result) == 3:
                        best_t, best_len, hist = result
                    elif len(result) == 4:
                        best_t, best_len, hist, conv_it = result
                    elif len(result) >= 5:
                        best_t, best_len, hist, conv_it, best_found_iter = result[:5]
                else:
                    best_t, best_len, hist = result
                t1 = time.time()
                row = {**cfg}
                row.update({
                    "seed": seed,
                    "best_len": float(best_len),
                    "convergence_iteration": int(conv_it) if conv_it is not None else None,
                    "best_found_iteration": int(best_found_iter) if best_found_iter is not None else None,
                    "time_s": t1 - t0,
                })
                writer.writerow(row)


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
    sweep(params)

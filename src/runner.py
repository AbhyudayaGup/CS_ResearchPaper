import argparse
import json
import time
import os
from .utils import generate_cities
from .aco import AntColony
import numpy as np


def run_once(args):
    coords = generate_cities(args.num_cities, seed=args.seed, clustered=args.clustered)
    ac = AntColony(coords, n_ants=args.num_ants, n_iterations=args.iterations,
                   alpha=args.alpha, beta=args.beta, rho=args.rho, Q=args.Q,
                   apply_two_opt=args.two_opt, seed=args.seed)
    start = time.time()
    result = ac.run(return_stats=True)
    # support older and newer return signatures (3,4,5+ values)
    best_tour = best_len = history = convergence_iteration = best_found_iter = stats = None
    if isinstance(result, tuple):
        if len(result) == 3:
            best_tour, best_len, history = result
        elif len(result) == 4:
            best_tour, best_len, history, convergence_iteration = result
        elif len(result) >= 5:
            best_tour, best_len, history, convergence_iteration, best_found_iter = result[:5]
        if len(result) >= 6:
            stats = result[5]
    else:
        best_tour, best_len, history = result
    elapsed = time.time() - start
    out = {
        "num_cities": args.num_cities,
        "num_ants": args.num_ants,
        "iterations": args.iterations,
        "alpha": args.alpha,
        "beta": args.beta,
        "rho": args.rho,
        "Q": args.Q,
        "seed": args.seed,
        "best_len": float(best_len),
        "convergence_iteration": int(convergence_iteration) if convergence_iteration is not None else None,
        "best_found_iteration": int(best_found_iter) if best_found_iter is not None else None,
        "time_s": elapsed,
    }
    if stats:
        out.update(
            {
                "convergence_time_s": float(stats.get("convergence_time_s", elapsed)),
                "objective_evals_to_convergence": int(stats.get("objective_evals_to_convergence", 0)),
                "objective_evals_total": int(stats.get("objective_evals_total", 0)),
                "run_time_s": float(stats.get("run_time_s", elapsed)),
            }
        )
    os.makedirs("results", exist_ok=True)
    fname = os.path.join("results", f"run_seed_{args.seed or 0}.json")
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved:", fname)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-cities", type=int, default=20)
    p.add_argument("--num-ants", type=int, default=None)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=5.0)
    p.add_argument("--rho", type=float, default=0.5)
    p.add_argument("--Q", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--two-opt", dest="two_opt", action="store_true")
    p.add_argument("--clustered", dest="clustered", action="store_true")
    args = p.parse_args()
    run_once(args)


if __name__ == "__main__":
    main()

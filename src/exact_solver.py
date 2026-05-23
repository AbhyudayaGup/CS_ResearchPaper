import itertools
import math
import time
from typing import Dict, List

import numpy as np


def _tour_length(tour: List[int], coords: np.ndarray) -> float:
    total = 0.0
    n = len(tour)
    for i in range(n):
        a = coords[tour[i]]
        b = coords[tour[(i + 1) % n]]
        total += math.hypot(a[0] - b[0], a[1] - b[1])
    return total


def _bruteforce_tsp(coords: np.ndarray) -> Dict:
    n = len(coords)
    nodes = list(range(n))
    best_tour = None
    best_len = float("inf")
    for perm in itertools.permutations(nodes[1:]):
        tour = [0] + list(perm)
        length = _tour_length(tour, coords)
        if length < best_len:
            best_len = length
            best_tour = tour
    return {
        "tour": best_tour,
        "length": float(best_len),
        "is_optimal": True,
        "method": "bruteforce",
        "status": "OPTIMAL",
    }


def _cp_sat_tsp(coords: np.ndarray, max_seconds: int = 120) -> Dict:
    try:
        from ortools.sat.python import cp_model
    except Exception as exc:
        raise RuntimeError(
            "ortools is required for exact solving beyond small instances. Install with: pip install ortools"
        ) from exc

    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    euclid = np.hypot(diff[..., 0], diff[..., 1])
    scaled = np.rint(euclid * 1000.0).astype(int)

    model = cp_model.CpModel()
    edge = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            edge[(i, j)] = model.NewBoolVar(f"x_{i}_{j}")

    for i in range(n):
        model.Add(sum(edge[(i, j)] for j in range(n) if j != i) == 1)
        model.Add(sum(edge[(j, i)] for j in range(n) if j != i) == 1)

    arcs = [(i, j, edge[(i, j)]) for i in range(n) for j in range(n) if i != j]
    model.AddCircuit(arcs)
    model.Minimize(sum(scaled[i, j] * edge[(i, j)] for i in range(n) for j in range(n) if i != j))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(max_seconds)
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("Exact solver failed to find a tour in the configured time limit.")

    succ = {}
    for i in range(n):
        for j in range(n):
            if i != j and solver.Value(edge[(i, j)]) == 1:
                succ[i] = j
                break

    tour = [0]
    seen = {0}
    cur = 0
    for _ in range(n - 1):
        nxt = succ[cur]
        if nxt in seen:
            break
        tour.append(nxt)
        seen.add(nxt)
        cur = nxt

    if len(tour) != n:
        raise RuntimeError("Solver returned an invalid cycle.")

    return {
        "tour": tour,
        "length": float(_tour_length(tour, coords)),
        "is_optimal": status == cp_model.OPTIMAL,
        "method": "cp_sat",
        "status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
    }


def solve_tsp_exact(coords: np.ndarray, max_seconds: int = 120) -> Dict:
    coords = np.asarray(coords)
    n = len(coords)
    if n < 3:
        raise ValueError("TSP requires at least 3 cities.")
    if n > 50:
        raise ValueError("This exact solver is capped at 50 cities for now.")

    t0 = time.time()
    if n <= 11:
        result = _bruteforce_tsp(coords)
    else:
        result = _cp_sat_tsp(coords, max_seconds=max_seconds)
    result["solve_time_s"] = float(time.time() - t0)
    return result

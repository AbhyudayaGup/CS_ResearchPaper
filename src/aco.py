import numpy as np
from .utils import distance_matrix
import math
import time


def two_opt_swap(tour, i, k):
    new = tour[:i] + tour[i:k + 1][::-1] + tour[k + 1:]
    return new


def two_opt(tour, coords, return_eval_count=False):
    improved = True
    best = tour
    best_len = tour_length(best, coords)
    eval_count = 1
    n = len(tour)
    while improved:
        improved = False
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                new = two_opt_swap(best, i, k)
                new_len = tour_length(new, coords)
                eval_count += 1
                if new_len < best_len:
                    best = new
                    best_len = new_len
                    improved = True
        # loop until no improvement
    if return_eval_count:
        return best, best_len, eval_count
    return best, best_len


def tour_length(tour, coords):
    coords = np.asarray(coords)
    n = len(tour)
    total = 0.0
    for i in range(n):
        a = coords[tour[i]]
        b = coords[tour[(i + 1) % n]]
        total += math.hypot(a[0] - b[0], a[1] - b[1])
    return total


class AntColony:
    def __init__(self, coords, n_ants=None, n_iterations=200, alpha=1.0, beta=5.0, rho=0.5, Q=100.0,
                 pheromone_init=None, apply_two_opt=True, seed=None, elitist_weight=0.0):
        self.coords = np.asarray(coords)
        self.n = len(coords)
        self.dist = distance_matrix(self.coords)
        self.heuristic = np.zeros_like(self.dist)
        with np.errstate(divide='ignore'):
            self.heuristic = 1.0 / (self.dist + 1e-12)
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.n_ants = n_ants or self.n
        self.n_iterations = n_iterations
        self.apply_two_opt = apply_two_opt
        self.elitist_weight = elitist_weight
        self.rng = np.random.default_rng(seed)
        if pheromone_init is None:
            pheromone_init = 1.0 / self.n
        self.pheromone = np.full((self.n, self.n), pheromone_init, dtype=float)

    def _select_next(self, current, visited_mask):
        pher = self.pheromone[current] ** self.alpha
        heur = self.heuristic[current] ** self.beta
        prob = pher * heur
        prob = prob * (~visited_mask)
        total = prob.sum()
        if total <= 0:
            choices = np.where(~visited_mask)[0]
            return int(self.rng.choice(choices))
        prob = prob / total
        return int(self.rng.choice(self.n, p=prob))

    def _construct_solutions(self):
        tours = []
        lengths = []
        eval_count = 0
        for _ in range(self.n_ants):
            start = int(self.rng.integers(0, self.n))
            tour = [start]
            visited = np.zeros(self.n, dtype=bool)
            visited[start] = True
            current = start
            while len(tour) < self.n:
                nxt = self._select_next(current, visited)
                tour.append(nxt)
                visited[nxt] = True
                current = nxt
            tours.append(tour)
            lengths.append(tour_length(tour, self.coords))
            eval_count += 1
        return tours, lengths, eval_count

    def _update_pheromones(self, tours, lengths, best_so_far=None, best_len=None):
        # evaporation
        self.pheromone *= (1.0 - self.rho)
        # deposit from each ant
        for tour, L in zip(tours, lengths):
            delta = self.Q / (L + 1e-12)
            for i in range(len(tour)):
                a = tour[i]
                b = tour[(i + 1) % len(tour)]
                self.pheromone[a, b] += delta
                self.pheromone[b, a] += delta
        # elitist deposit
        if best_so_far is not None and self.elitist_weight > 0 and best_len is not None:
            delta = self.elitist_weight * self.Q / (best_len + 1e-12)
            for i in range(len(best_so_far)):
                a = best_so_far[i]
                b = best_so_far[(i + 1) % len(best_so_far)]
                self.pheromone[a, b] += delta
                self.pheromone[b, a] += delta

    def run(self, callback=None, return_stats=False, target_length=None, target_gap_pct=0.0, target_tolerance=1e-9):
        best_tour = None
        best_len = float('inf')
        history = []
        convergence_iteration = None
        initial_worst = float('inf')
        best_found_iter = None
        stopped_early = False
        t0 = time.perf_counter()
        objective_evals_total = 0
        objective_evals_to_convergence = None
        convergence_time_s = None

        def reached_target(current_best: float) -> bool:
            if target_length is None:
                return False
            target = float(target_length)
            if target <= 0:
                tolerance = max(float(target_tolerance), 1e-9)
                return abs(float(current_best) - target) <= tolerance
            gap_pct = ((float(current_best) - target) / target) * 100.0
            tolerance = max(float(target_tolerance), 1e-6)
            if abs(gap_pct) <= tolerance:
                return True
            return gap_pct <= float(target_gap_pct)

        for it in range(self.n_iterations):
            tours, lengths, eval_count = self._construct_solutions()
            objective_evals_total += eval_count
            # optional local search
            if self.apply_two_opt:
                new_tours = []
                new_lengths = []
                for tour, L in zip(tours, lengths):
                    t, l, two_opt_evals = two_opt(tour, self.coords, return_eval_count=True)
                    objective_evals_total += two_opt_evals
                    new_tours.append(t)
                    new_lengths.append(l)
                tours, lengths = new_tours, new_lengths
            # track initial worst tour
            if it == 0:
                initial_worst = max(lengths)
            # update best
            prev_best = best_len
            for tour, L in zip(tours, lengths):
                if L < best_len:
                    best_len = L
                    best_tour = tour
                    best_found_iter = it
                    objective_evals_to_convergence = objective_evals_total
                    convergence_time_s = time.perf_counter() - t0
            # mark convergence at first improvement
            if best_len < prev_best and convergence_iteration is None:
                convergence_iteration = it
            # update pheromones
            self._update_pheromones(tours, lengths, best_so_far=best_tour, best_len=best_len)
            history.append(best_len)
            if callback is not None:
                callback(iteration=it, best_len=best_len, best_tour=best_tour, convergence_iteration=convergence_iteration)

            if reached_target(best_len):
                stopped_early = True
                break
        if convergence_iteration is None:
            convergence_iteration = len(history) - 1 if history else 0
        if best_found_iter is None:
            best_found_iter = len(history) - 1 if history else 0
            objective_evals_to_convergence = objective_evals_total
            convergence_time_s = time.perf_counter() - t0

        total_time_s = time.perf_counter() - t0
        stats = {
            "run_time_s": float(total_time_s),
            "convergence_time_s": float(convergence_time_s),
            "objective_evals_total": int(objective_evals_total),
            "objective_evals_to_convergence": int(objective_evals_to_convergence),
            "convergence_iteration": int(convergence_iteration),
            "best_found_iteration": int(best_found_iter),
            "iterations_executed": int(len(history)),
            "stopped_early": bool(stopped_early),
        }

        if return_stats:
            return best_tour, best_len, history, convergence_iteration, best_found_iter, stats
        return best_tour, best_len, history, convergence_iteration, best_found_iter

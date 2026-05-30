import numpy as np
from .utils import distance_matrix
from . import dynamic_env
import math
import time


def two_opt_swap(tour, i, k):
    new = tour[:i] + tour[i:k + 1][::-1] + tour[k + 1:]
    return new


def two_opt(tour, coords, return_eval_count=False, blocked_mask=None):
    improved = True
    best = tour
    best_len = tour_length(best, coords, blocked_mask=blocked_mask)
    eval_count = 1
    n = len(tour)
    while improved:
        improved = False
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                new = two_opt_swap(best, i, k)
                if blocked_mask is not None and dynamic_env.tour_has_blocked_edge(new, blocked_mask):
                    continue
                new_len = tour_length(new, coords, blocked_mask=blocked_mask)
                eval_count += 1
                if new_len < best_len:
                    best = new
                    best_len = new_len
                    improved = True
        # loop until no improvement
    if return_eval_count:
        return best, best_len, eval_count
    return best, best_len


def tour_length(tour, coords, blocked_mask=None):
    coords = np.asarray(coords)
    n = len(tour)
    total = 0.0
    for i in range(n):
        if blocked_mask is not None and bool(blocked_mask[int(tour[i]), int(tour[(i + 1) % n])]):
            return float('inf')
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

    def _select_next(self, current, visited_mask, blocked_mask=None):
        pher = self.pheromone[current] ** self.alpha
        heur = self.heuristic[current] ** self.beta
        prob = pher * heur
        allowed = ~visited_mask
        if blocked_mask is not None:
            allowed = allowed & (~blocked_mask[current])
        prob = prob * allowed
        total = prob.sum()
        if total <= 0:
            choices = np.where(allowed)[0]
            if len(choices) == 0 and blocked_mask is not None:
                choices = np.where(~visited_mask)[0]
            if len(choices) == 0:
                choices = np.arange(self.n)
            return int(self.rng.choice(choices))
        prob = prob / total
        return int(self.rng.choice(self.n, p=prob))

    def _construct_solutions(self, blocked_mask=None):
        tours = []
        lengths = []
        eval_count = 0
        for _ in range(self.n_ants):
            valid_tour = dynamic_env.random_valid_tour(self.n, self.rng, blocked_mask)
            if valid_tour is None:
                tours.append(None)
                lengths.append(float('inf'))
                continue
            # build tour through pheromone-guided selection while avoiding blocked edges
            start = int(valid_tour[0])
            tour = [start]
            visited = np.zeros(self.n, dtype=bool)
            visited[start] = True
            current = start
            while len(tour) < self.n:
                nxt = self._select_next(current, visited, blocked_mask=blocked_mask)
                if visited[nxt] or (blocked_mask is not None and blocked_mask[current, nxt]):
                    # fallback to a valid unused city if possible
                    choices = np.where((~visited) & ((~blocked_mask[current]) if blocked_mask is not None else True))[0]
                    if len(choices) == 0:
                        choices = np.where(~visited)[0]
                    nxt = int(self.rng.choice(choices))
                tour.append(nxt)
                visited[nxt] = True
                current = nxt
            if dynamic_env.tour_has_blocked_edge(tour, blocked_mask):
                lengths.append(float('inf'))
            else:
                lengths.append(tour_length(tour, self.coords, blocked_mask=blocked_mask))
            tours.append(tour)
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
        # quick feasibility check: ensure at least one valid tour exists under the initial mask
        try:
            blocked_mask0 = dynamic_env.get_block_mask(self.n, iteration=0)
            if dynamic_env.random_valid_tour(self.n, self.rng, blocked_mask0) is None:
                raise RuntimeError("Blocked edges make this TSP infeasible to build a valid Hamiltonian tour. Reduce blocked edges or change the seed.")
        except Exception:
            # if dynamic_env is unavailable for some reason, proceed and let later checks catch issues
            pass
        objective_evals_total = 0
        objective_evals_to_convergence = None
        convergence_time_s = None

        def reached_target(current_best: float) -> bool:
            # In dynamic mode we never consider convergence/early stop because
            # the environment changes during the run.
            try:
                from . import dynamic_env
                if getattr(dynamic_env, "MODE", "standard") == "dynamic":
                    return False
            except Exception:
                pass
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
            blocked_mask = dynamic_env.get_block_mask(self.n, iteration=it)
            # If environment changed, invalidate previously best tour if it uses blocked edges
            try:
                if getattr(dynamic_env, "MODE", "standard") == "dynamic":
                    if best_tour is not None and dynamic_env.tour_has_blocked_edge(best_tour, blocked_mask):
                        best_tour = None
                        best_len = float('inf')
            except Exception:
                pass
            tours, lengths, eval_count = self._construct_solutions(blocked_mask=blocked_mask)
            objective_evals_total += eval_count
            # compute valid fraction for diagnostics
            valid_total = len(lengths) if lengths else 0
            valid_count = sum(1 for L in lengths if np.isfinite(L)) if valid_total > 0 else 0
            valid_fraction = float(valid_count) / float(max(1, valid_total))
            # optional local search
            if self.apply_two_opt:
                pre_tours, pre_lengths = tours, lengths
                new_tours = []
                new_lengths = []
                for tour, L in zip(tours, lengths):
                    if tour is None or not np.isfinite(L):
                        continue
                    t, l, two_opt_evals = two_opt(tour, self.coords, return_eval_count=True, blocked_mask=blocked_mask)
                    if dynamic_env.tour_has_blocked_edge(t, blocked_mask):
                        l = float('inf')
                    objective_evals_total += two_opt_evals
                    new_tours.append(t)
                    new_lengths.append(l)
                # If local search filtered out all candidates, fall back to pre-search results
                if not new_lengths:
                    tours, lengths = pre_tours, pre_lengths
                else:
                    tours, lengths = new_tours, new_lengths
            # track initial worst tour
            if it == 0:
                initial_worst = max(lengths) if lengths else float('inf')
            # update best
            prev_best = best_len
            for tour, L in zip(tours, lengths):
                if tour is None or not np.isfinite(L):
                    continue
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
                callback(iteration=it, best_len=best_len, best_tour=best_tour, convergence_iteration=convergence_iteration, valid_fraction=valid_fraction)

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

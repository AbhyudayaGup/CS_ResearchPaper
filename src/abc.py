from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from .aco import two_opt, tour_length
from . import dynamic_env


def _random_tour(rng: np.random.Generator, n: int, blocked_mask=None) -> list[int] | None:
    return dynamic_env.random_valid_tour(n, rng, blocked_mask)


def _neighbor_tour(rng: np.random.Generator, tour: list[int], blocked_mask=None) -> list[int] | None:
    candidate = tour.copy()
    n = len(candidate)
    if n < 2:
        return candidate
    i, k = sorted(rng.choice(n, size=2, replace=False).tolist())
    if i == k:
        return candidate
    candidate[i : k + 1] = reversed(candidate[i : k + 1])
    if blocked_mask is not None and dynamic_env.tour_has_blocked_edge(candidate, blocked_mask):
        return None
    return candidate


def _evaluate_tour(coords: np.ndarray, tour: list[int] | None, apply_two_opt: bool, blocked_mask=None) -> tuple[list[int] | None, float, int]:
    if tour is None:
        return None, float("inf"), 0
    if blocked_mask is not None and dynamic_env.tour_has_blocked_edge(tour, blocked_mask):
        tour = dynamic_env.random_valid_tour(len(coords), np.random.default_rng(), blocked_mask)
        if tour is None:
            return None, float("inf"), 0
    if apply_two_opt:
        improved_tour, improved_len, eval_count = two_opt(tour, coords, return_eval_count=True, blocked_mask=blocked_mask)
        return improved_tour, float(improved_len), int(eval_count)
    return tour, float(tour_length(tour, coords, blocked_mask=blocked_mask)), 1


class ArtificialBeeColony:
    def __init__(
        self,
        coords,
        n_food_sources: int | None = None,
        n_iterations: int = 200,
        limit: int | None = None,
        apply_two_opt: bool = True,
        seed: int | None = None,
    ):
        self.coords = np.asarray(coords)
        self.n = len(coords)
        self.n_food_sources = int(n_food_sources or self.n)
        self.n_iterations = int(n_iterations)
        self.limit = int(limit if limit is not None else max(5, self.n_food_sources * 3))
        self.apply_two_opt = bool(apply_two_opt)
        self.rng = np.random.default_rng(seed)

    def _fitness(self, lengths: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.asarray(lengths, dtype=float))

    def _select_source(self, probabilities: np.ndarray) -> int:
        return int(self.rng.choice(len(probabilities), p=probabilities))

    def run(
        self,
        callback: Callable[..., None] | None = None,
        return_stats: bool = False,
        target_length: float | None = None,
        target_gap_pct: float = 0.0,
        target_tolerance: float = 1e-9,
    ):
        best_tour: list[int] | None = None
        best_len = float("inf")
        history: list[float] = []
        convergence_iteration: int | None = None
        best_found_iter: int | None = None
        stopped_early = False
        objective_evals_total = 0
        objective_evals_to_convergence: int | None = None
        convergence_time_s: float | None = None
        t0 = time.perf_counter()

        def reached_target(current_best: float) -> bool:
            # disable early stopping for truly dynamic environments
            try:
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

        food_sources = []
        source_lengths = []
        trials = np.zeros(self.n_food_sources, dtype=int)

        blocked_mask = dynamic_env.get_block_mask(self.n, iteration=0)

        for _ in range(self.n_food_sources):
            tour = _random_tour(self.rng, self.n, blocked_mask)
            tour, length, eval_count = _evaluate_tour(self.coords, tour, self.apply_two_opt, blocked_mask=blocked_mask)
            objective_evals_total += eval_count
            food_sources.append(tour)
            source_lengths.append(length)
            if length < best_len:
                best_len = float(length)
                best_tour = tour.copy()
                best_found_iter = 0
                objective_evals_to_convergence = objective_evals_total
                convergence_time_s = time.perf_counter() - t0

        # If initial population contains no feasible solutions, fail fast with clear message
        if all((not np.isfinite(float(l)) for l in source_lengths)):
            raise RuntimeError("No feasible tours could be generated under the configured blocked edges. Adjust blocked edge settings or seed.")

        if best_tour is not None:
            convergence_iteration = 0

        if reached_target(best_len):
            stopped_early = True
        else:
            for it in range(self.n_iterations):
                blocked_mask = dynamic_env.get_block_mask(self.n, iteration=it)
                # In dynamic environments, if the current best tour becomes invalid, reset it so search can adapt
                try:
                    if getattr(dynamic_env, "MODE", "standard") == "dynamic":
                        if best_tour is not None and dynamic_env.tour_has_blocked_edge(best_tour, blocked_mask):
                            best_tour = None
                            best_len = float('inf')
                except Exception:
                    pass
                lengths_arr = np.asarray(source_lengths, dtype=float)
                fitness = self._fitness(lengths_arr)
                total_fitness = float(fitness.sum())
                if total_fitness <= 0:
                    probabilities = np.full(self.n_food_sources, 1.0 / float(self.n_food_sources))
                else:
                    probabilities = fitness / total_fitness

                # Employed bees.
                for idx in range(self.n_food_sources):
                    candidate = _neighbor_tour(self.rng, food_sources[idx], blocked_mask=blocked_mask)
                    candidate, candidate_len, eval_count = _evaluate_tour(self.coords, candidate, self.apply_two_opt, blocked_mask=blocked_mask)
                    objective_evals_total += eval_count
                    if candidate_len < source_lengths[idx]:
                        food_sources[idx] = candidate
                        source_lengths[idx] = float(candidate_len)
                        trials[idx] = 0
                    else:
                        trials[idx] += 1

                # Onlooker bees.
                for _ in range(self.n_food_sources):
                    idx = self._select_source(probabilities)
                    candidate = _neighbor_tour(self.rng, food_sources[idx], blocked_mask=blocked_mask)
                    candidate, candidate_len, eval_count = _evaluate_tour(self.coords, candidate, self.apply_two_opt, blocked_mask=blocked_mask)
                    objective_evals_total += eval_count
                    if candidate_len < source_lengths[idx]:
                        food_sources[idx] = candidate
                        source_lengths[idx] = float(candidate_len)
                        trials[idx] = 0
                    else:
                        trials[idx] += 1

                # Scout bees.
                for idx in range(self.n_food_sources):
                    if trials[idx] < self.limit:
                        continue
                    tour = _random_tour(self.rng, self.n, blocked_mask)
                    tour, length, eval_count = _evaluate_tour(self.coords, tour, self.apply_two_opt, blocked_mask=blocked_mask)
                    objective_evals_total += eval_count
                    food_sources[idx] = tour
                    source_lengths[idx] = float(length)
                    trials[idx] = 0

                prev_best = best_len
                for tour, length in zip(food_sources, source_lengths):
                    if length < best_len:
                        best_len = float(length)
                        best_tour = tour.copy()
                        best_found_iter = it
                        objective_evals_to_convergence = objective_evals_total
                        convergence_time_s = time.perf_counter() - t0

                if best_len < prev_best and convergence_iteration is None:
                    convergence_iteration = it

                # diagnostics: fraction of valid food sources
                total = len(source_lengths)
                valid = sum(1 for L in source_lengths if np.isfinite(L)) if total > 0 else 0
                valid_fraction = float(valid) / float(max(1, total))
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
        stats: dict[str, Any] = {
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
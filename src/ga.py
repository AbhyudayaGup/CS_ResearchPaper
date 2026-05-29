from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from .aco import two_opt, tour_length


def _random_tour(rng: np.random.Generator, n: int) -> list[int]:
    return rng.permutation(n).astype(int).tolist()


def _evaluate_tour(coords: np.ndarray, tour: list[int], apply_two_opt: bool) -> tuple[list[int], float, int]:
    if apply_two_opt:
        improved_tour, improved_len, eval_count = two_opt(tour, coords, return_eval_count=True)
        return improved_tour, float(improved_len), int(eval_count)
    return tour, float(tour_length(tour, coords)), 1


def _order_crossover(rng: np.random.Generator, parent_a: list[int], parent_b: list[int]) -> list[int]:
    n = len(parent_a)
    if n < 2:
        return parent_a.copy()
    i, j = sorted(rng.choice(n, size=2, replace=False).tolist())
    child = [-1] * n
    child[i : j + 1] = parent_a[i : j + 1]
    used = set(child[i : j + 1])
    fill_index = (j + 1) % n
    for gene in parent_b:
        if gene in used:
            continue
        while child[fill_index] != -1:
            fill_index = (fill_index + 1) % n
        child[fill_index] = int(gene)
        used.add(int(gene))
    return child


def _mutate_tour(rng: np.random.Generator, tour: list[int]) -> list[int]:
    candidate = tour.copy()
    n = len(candidate)
    if n < 2:
        return candidate
    i, j = sorted(rng.choice(n, size=2, replace=False).tolist())
    if rng.random() < 0.5:
        candidate[i], candidate[j] = candidate[j], candidate[i]
    else:
        candidate[i : j + 1] = reversed(candidate[i : j + 1])
    return candidate


class GeneticAlgorithm:
    def __init__(
        self,
        coords,
        n_population: int | None = None,
        n_iterations: int = 200,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.2,
        elite_fraction: float = 0.1,
        tournament_size: int = 3,
        apply_two_opt: bool = True,
        seed: int | None = None,
    ):
        self.coords = np.asarray(coords)
        self.n = len(coords)
        self.n_population = int(n_population or self.n)
        self.n_iterations = int(n_iterations)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.elite_count = max(1, int(round(self.n_population * float(elite_fraction))))
        self.tournament_size = max(2, int(tournament_size))
        self.apply_two_opt = bool(apply_two_opt)
        self.rng = np.random.default_rng(seed)

    def _select_parent(self, population: list[list[int]], scores: list[float]) -> list[int]:
        tournament_size = min(self.tournament_size, len(population))
        contenders = self.rng.choice(len(population), size=tournament_size, replace=False)
        best_idx = int(contenders[0])
        best_score = float(scores[best_idx])
        for idx in contenders[1:]:
            score = float(scores[int(idx)])
            if score < best_score:
                best_idx = int(idx)
                best_score = score
        return population[best_idx].copy()

    def _breed_child(self, parent_a: list[int], parent_b: list[int]) -> list[int]:
        if self.rng.random() < self.crossover_rate:
            child = _order_crossover(self.rng, parent_a, parent_b)
        else:
            child = parent_a.copy()
        if self.rng.random() < self.mutation_rate:
            child = _mutate_tour(self.rng, child)
        return child

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

        population: list[list[int]] = []
        scores: list[float] = []
        for _ in range(self.n_population):
            tour = _random_tour(self.rng, self.n)
            tour, length, eval_count = _evaluate_tour(self.coords, tour, self.apply_two_opt)
            objective_evals_total += eval_count
            population.append(tour)
            scores.append(float(length))
            if length < best_len:
                best_len = float(length)
                best_tour = tour.copy()
                best_found_iter = 0
                objective_evals_to_convergence = objective_evals_total
                convergence_time_s = time.perf_counter() - t0

        if best_tour is not None:
            convergence_iteration = 0

        if reached_target(best_len):
            stopped_early = True
        else:
            for it in range(self.n_iterations):
                ranked = sorted(zip(population, scores), key=lambda item: float(item[1]))
                elites = [tour.copy() for tour, _ in ranked[: self.elite_count]]
                elite_scores = [float(score) for _, score in ranked[: self.elite_count]]

                new_population: list[list[int]] = elites
                new_scores: list[float] = elite_scores

                while len(new_population) < self.n_population:
                    parent_a = self._select_parent(population, scores)
                    parent_b = self._select_parent(population, scores)
                    child = self._breed_child(parent_a, parent_b)
                    child, child_len, eval_count = _evaluate_tour(self.coords, child, self.apply_two_opt)
                    objective_evals_total += eval_count
                    new_population.append(child)
                    new_scores.append(float(child_len))

                population = new_population
                scores = new_scores

                prev_best = best_len
                for tour, score in zip(population, scores):
                    if score < best_len:
                        best_len = float(score)
                        best_tour = tour.copy()
                        best_found_iter = it
                        objective_evals_to_convergence = objective_evals_total
                        convergence_time_s = time.perf_counter() - t0

                if best_len < prev_best and convergence_iteration is None:
                    convergence_iteration = it

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
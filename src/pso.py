import numpy as np
import math
import time
from .utils import distance_matrix
from .aco import two_opt, tour_length


class ParticleSwarm:
    def __init__(self, coords, n_particles=None, n_iterations=200, w=0.5, c1=1.5, c2=1.5,
                 apply_two_opt=True, seed=None):
        self.coords = np.asarray(coords)
        self.n = len(coords)
        self.dist = distance_matrix(self.coords)
        self.n_particles = n_particles or self.n
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.apply_two_opt = apply_two_opt
        self.rng = np.random.default_rng(seed)

        # Initialize particles: positions are real-valued priority vectors
        self.positions = self.rng.random((self.n_particles, self.n))
        self.velocities = np.zeros((self.n_particles, self.n))

        # personal best positions and scores
        self.pbest_pos = self.positions.copy()
        self.pbest_score = np.full(self.n_particles, float('inf'))

        # global best
        self.gbest_pos = None
        self.gbest_score = float('inf')

    def _decode(self, position):
        # decode priority vector into permutation: higher value -> earlier in tour
        # tie-break with small noise
        noise = self.rng.random(len(position)) * 1e-9
        keys = position + noise
        perm = np.argsort(-keys)
        return list(map(int, perm))

    def _evaluate(self, position):
        tour = self._decode(position)
        if self.apply_two_opt:
            tour, L, eval_count = two_opt(tour, self.coords, return_eval_count=True)
            return tour, L, eval_count
        L = tour_length(tour, self.coords)
        return tour, L, 1

    def run(self, callback=None, return_stats=False, target_length=None, target_gap_pct=0.0, target_tolerance=1e-9):
        best_tour = None
        best_len = float('inf')
        history = []
        convergence_iteration = None
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

        # evaluate initial particles
        for i in range(self.n_particles):
            tour, L, eval_count = self._evaluate(self.positions[i])
            objective_evals_total += eval_count
            self.pbest_pos[i] = self.positions[i].copy()
            self.pbest_score[i] = L
            if L < self.gbest_score:
                self.gbest_score = L
                self.gbest_pos = self.positions[i].copy()
                best_tour = tour
                best_len = L
                best_found_iter = 0
                objective_evals_to_convergence = objective_evals_total
                convergence_time_s = time.perf_counter() - t0
                if reached_target(best_len):
                    stopped_early = True
                    break

        if not stopped_early:
            for it in range(self.n_iterations):
                for i in range(self.n_particles):
                    r1 = self.rng.random(self.n)
                    r2 = self.rng.random(self.n)
                    self.velocities[i] = (self.w * self.velocities[i] +
                                           self.c1 * r1 * (self.pbest_pos[i] - self.positions[i]) +
                                           self.c2 * r2 * (self.gbest_pos - self.positions[i]))
                    self.positions[i] = self.positions[i] + self.velocities[i]
                    # keep positions bounded
                    self.positions[i] = np.mod(self.positions[i], 1.0)

                    tour, L, eval_count = self._evaluate(self.positions[i])
                    objective_evals_total += eval_count
                    if L < self.pbest_score[i]:
                        self.pbest_score[i] = L
                        self.pbest_pos[i] = self.positions[i].copy()
                    if L < self.gbest_score:
                        prev_best = self.gbest_score
                        self.gbest_score = L
                        self.gbest_pos = self.positions[i].copy()
                        best_tour = tour
                        best_len = L
                        best_found_iter = it
                        objective_evals_to_convergence = objective_evals_total
                        convergence_time_s = time.perf_counter() - t0
                        if convergence_iteration is None and self.gbest_score < prev_best:
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

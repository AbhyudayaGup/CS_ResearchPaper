import numpy as np
import math
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
        L = tour_length(tour, self.coords)
        if self.apply_two_opt:
            tour, L = two_opt(tour, self.coords)
        return tour, L

    def run(self, callback=None):
        best_tour = None
        best_len = float('inf')
        history = []
        convergence_iteration = None

        # evaluate initial particles
        for i in range(self.n_particles):
            tour, L = self._evaluate(self.positions[i])
            self.pbest_pos[i] = self.positions[i].copy()
            self.pbest_score[i] = L
            if L < self.gbest_score:
                self.gbest_score = L
                self.gbest_pos = self.positions[i].copy()
                best_tour = tour
                best_len = L

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

                tour, L = self._evaluate(self.positions[i])
                if L < self.pbest_score[i]:
                    self.pbest_score[i] = L
                    self.pbest_pos[i] = self.positions[i].copy()
                if L < self.gbest_score:
                    prev_best = self.gbest_score
                    self.gbest_score = L
                    self.gbest_pos = self.positions[i].copy()
                    best_tour = tour
                    best_len = L
                    if convergence_iteration is None and self.gbest_score < prev_best:
                        convergence_iteration = it

            history.append(best_len)
            if callback is not None:
                callback(iteration=it, best_len=best_len, best_tour=best_tour, convergence_iteration=convergence_iteration)

        if convergence_iteration is None:
            convergence_iteration = self.n_iterations - 1

        # find first iteration where best_len achieved
        best_found_iter = None
        for idx, val in enumerate(history):
            if abs(val - best_len) < 1e-12:
                best_found_iter = idx
                break
        if best_found_iter is None:
            best_found_iter = self.n_iterations - 1

        return best_tour, best_len, history, convergence_iteration, best_found_iter

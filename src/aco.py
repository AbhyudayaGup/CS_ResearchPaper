import numpy as np
from .utils import distance_matrix
import math


def two_opt_swap(tour, i, k):
    new = tour[:i] + tour[i:k + 1][::-1] + tour[k + 1:]
    return new


def two_opt(tour, coords):
    improved = True
    best = tour
    best_len = tour_length(best, coords)
    n = len(tour)
    while improved:
        improved = False
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                new = two_opt_swap(best, i, k)
                new_len = tour_length(new, coords)
                if new_len < best_len:
                    best = new
                    best_len = new_len
                    improved = True
        # loop until no improvement
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
        return tours, lengths

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

    def run(self, callback=None):
        best_tour = None
        best_len = float('inf')
        history = []
        convergence_iteration = None
        initial_worst = float('inf')
        for it in range(self.n_iterations):
            tours, lengths = self._construct_solutions()
            # optional local search
            if self.apply_two_opt:
                new_tours = []
                new_lengths = []
                for tour, L in zip(tours, lengths):
                    t, l = two_opt(tour, self.coords)
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
            # mark convergence at first improvement
            if best_len < prev_best and convergence_iteration is None:
                convergence_iteration = it
            # update pheromones
            self._update_pheromones(tours, lengths, best_so_far=best_tour, best_len=best_len)
            history.append(best_len)
            if callback is not None:
                callback(iteration=it, best_len=best_len, best_tour=best_tour, convergence_iteration=convergence_iteration)
        if convergence_iteration is None:
            convergence_iteration = self.n_iterations - 1
        # find the first iteration where the best_len was achieved
        best_found_iter = None
        for idx, val in enumerate(history):
            if abs(val - best_len) < 1e-12:
                best_found_iter = idx
                break
        if best_found_iter is None:
            best_found_iter = self.n_iterations - 1
        return best_tour, best_len, history, convergence_iteration, best_found_iter

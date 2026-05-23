import itertools
import math
from src.utils import generate_cities, distance_matrix
from src.aco import AntColony, tour_length


def brute_force_opt(coords):
    n = len(coords)
    idx = list(range(n))
    best = None
    best_len = float('inf')
    for perm in itertools.permutations(idx[1:]):
        tour = [idx[0]] + list(perm)
        L = tour_length(tour, coords)
        if L < best_len:
            best_len = L
            best = tour
    return best_len, best


def test_distance_matrix_properties():
    coords = generate_cities(6, seed=1)
    D = distance_matrix(coords)
    assert D.shape == (6, 6)
    # diagonal zeros
    assert all(abs(D[i, i]) < 1e-12 for i in range(6))
    # symmetry
    for i in range(6):
        for j in range(6):
            assert abs(D[i, j] - D[j, i]) < 1e-9


def test_aco_finds_optimal_small():
    coords = generate_cities(8, seed=2)
    brute_len, _ = brute_force_opt(coords)
    ac = AntColony(coords, n_ants=8, n_iterations=500, alpha=1.0, beta=5.0, rho=0.5, Q=100.0, apply_two_opt=True, seed=42)
    best_t, best_len, hist, conv_it, best_found_iter = ac.run()
    # Allow a tiny tolerance, but expect near-optimal due to 2-opt and many iterations
    assert best_len <= brute_len * 1.02

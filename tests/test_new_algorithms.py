from __future__ import annotations

from src.abc import ArtificialBeeColony
from src.ga import GeneticAlgorithm
from src.utils import generate_cities


def test_abc_small_instance():
    coords = generate_cities(8, seed=11, clustered=False)
    abc = ArtificialBeeColony(coords, n_food_sources=8, n_iterations=30, limit=20, apply_two_opt=True, seed=3)
    best_tour, best_len, history, conv_iter, best_found = abc.run()
    assert best_tour is not None
    assert isinstance(best_len, float)
    assert len(best_tour) == len(coords)
    assert best_len > 0
    assert isinstance(history, list)


def test_ga_small_instance():
    coords = generate_cities(8, seed=12, clustered=False)
    ga = GeneticAlgorithm(
        coords,
        n_population=10,
        n_iterations=30,
        crossover_rate=0.9,
        mutation_rate=0.2,
        elite_fraction=0.2,
        tournament_size=3,
        apply_two_opt=True,
        seed=4,
    )
    best_tour, best_len, history, conv_iter, best_found = ga.run()
    assert best_tour is not None
    assert isinstance(best_len, float)
    assert len(best_tour) == len(coords)
    assert best_len > 0
    assert isinstance(history, list)

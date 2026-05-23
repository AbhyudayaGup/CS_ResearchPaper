import numpy as np
from src.utils import generate_cities
from src.pso import ParticleSwarm


def test_pso_small_instance():
    coords = generate_cities(8, seed=42, clustered=False)
    pso = ParticleSwarm(coords, n_particles=10, n_iterations=50, w=0.6, c1=1.2, c2=1.2, apply_two_opt=True, seed=1)
    best_tour, best_len, history, conv_iter, best_found = pso.run()
    assert best_tour is not None
    assert isinstance(best_len, float)
    assert len(best_tour) == len(coords)
    assert best_len > 0
from src.exact_solver import solve_tsp_exact
from src.utils import generate_cities


def test_exact_solver_small_instance_optimal():
    coords = generate_cities(8, seed=7)
    result = solve_tsp_exact(coords, max_seconds=30)
    assert result["tour"] is not None
    assert len(result["tour"]) == 8
    assert result["length"] > 0
    assert result["status"] == "OPTIMAL"

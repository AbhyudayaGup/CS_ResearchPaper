import numpy as np


def generate_cities(n, seed=None, clustered=False):
    """Generate n 2D Euclidean city coordinates.

    If clustered=True, generate a few clusters to make the instance interesting.
    """
    rng = np.random.default_rng(seed)
    if not clustered:
        coords = rng.random((n, 2)) * 100.0
    else:
        k = max(2, n // 10)
        centers = rng.random((k, 2)) * 100.0
        coords = []
        for i in range(n):
            c = centers[i % k]
            coords.append(c + rng.normal(scale=5.0, size=2))
        coords = np.vstack(coords)
    return coords


def distance_matrix(coords):
    """Return an (n,n) matrix of Euclidean distances for coords (n,2)."""
    coords = np.asarray(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.hypot(diff[..., 0], diff[..., 1])
    return D

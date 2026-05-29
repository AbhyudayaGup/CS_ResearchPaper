from __future__ import annotations

import numpy as np
from typing import Optional

# Global parameters for noisy/dynamic environments. Web UI will set these.
MODE = "standard"  # one of 'standard','noisy','dynamic'
BLOCKED_FRACTION = 0.05
BLOCKED_COUNT = None
PENALTY = 1e6
SEED = None


def set_params(mode: str = "standard", blocked_fraction: float = 0.05, blocked_count: int | None = None, penalty: float = 1e6, seed: int | None = None):
    global MODE, BLOCKED_FRACTION, BLOCKED_COUNT, PENALTY, SEED
    MODE = str(mode)
    BLOCKED_FRACTION = float(blocked_fraction)
    BLOCKED_COUNT = None if blocked_count is None else int(blocked_count)
    PENALTY = float(penalty)
    SEED = None if seed is None or int(seed) == 0 else int(seed)


def clear_params():
    set_params("standard", 0.0, None, 1e6, None)


def _random_block_mask(n: int, rng: np.random.Generator, count: int | None = None, fraction: float = 0.0) -> np.ndarray:
    # returns a symmetric penalty matrix with zeros on diag and penalty for blocked edges
    mask = np.zeros((n, n), dtype=bool)
    possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if count is None:
        count = int(max(0, min(len(possible), round(len(possible) * fraction))))
    count = max(0, min(len(possible), int(count)))
    if count == 0:
        return mask
    chosen = rng.choice(len(possible), size=count, replace=False)
    for idx in chosen:
        i, j = possible[int(idx)]
        mask[i, j] = True
        mask[j, i] = True
    return mask


def get_penalty_matrix(n: int, iteration: int | None = None) -> Optional[np.ndarray]:
    """Return a penalty matrix (n x n) or None for standard mode.

    - For 'noisy' mode the returned matrix is fixed given the seed (or random if seed None).
    - For 'dynamic' mode the returned matrix may change with iteration (uses iteration to reseed).
    """
    global MODE, BLOCKED_FRACTION, BLOCKED_COUNT, PENALTY, SEED
    if MODE == "standard":
        return None
    rng_seed = SEED if SEED is not None else None
    if MODE == "dynamic" and iteration is not None:
        # vary with iteration to simulate changing inaccessible edges
        rng_seed = (SEED or 0) + int(iteration) * 7919
    rng = np.random.default_rng(rng_seed)
    mask = _random_block_mask(n, rng, count=BLOCKED_COUNT, fraction=BLOCKED_FRACTION)
    if not mask.any():
        return None
    M = np.zeros((n, n), dtype=float)
    M[mask] = PENALTY
    return M

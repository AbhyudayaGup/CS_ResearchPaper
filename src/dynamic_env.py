from __future__ import annotations

import numpy as np
from typing import Optional

# Global parameters for noisy/dynamic environments. Web UI will set these.
MODE = "standard"  # one of 'standard','noisy','dynamic'
BLOCKED_FRACTION = 0.05
BLOCKED_COUNT = None
PENALTY = 1e6
SEED = None
AUTO_RELAX = False
LAST_RELAXED_UNBLOCKED = 0
# Cache the first generated noisy mask for a given instance configuration so
# noisy mode stays fixed instead of changing on every call.
NOISY_MASK_CACHE: dict[tuple, np.ndarray] = {}


def set_params(mode: str = "standard", blocked_fraction: float = 0.05, blocked_count: int | None = None, penalty: float = 1e6, seed: int | None = None, auto_relax: bool = False, **kwargs):
    """Set global dynamic/noisy TSP parameters.

    Accepts extra kwargs and ignores them to remain compatible with older UI calls.
    """
    global MODE, BLOCKED_FRACTION, BLOCKED_COUNT, PENALTY, SEED, AUTO_RELAX, NOISY_MASK_CACHE, LAST_RELAXED_UNBLOCKED
    MODE = str(mode)
    BLOCKED_FRACTION = float(blocked_fraction)
    BLOCKED_COUNT = None if blocked_count is None else int(blocked_count)
    PENALTY = float(penalty)
    SEED = None if seed is None or int(seed) == 0 else int(seed)
    AUTO_RELAX = bool(auto_relax)
    NOISY_MASK_CACHE = {}
    LAST_RELAXED_UNBLOCKED = 0


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


def get_block_mask(n: int, iteration: int | None = None) -> Optional[np.ndarray]:
    """Return a boolean mask of blocked edges or None for standard mode.

    - For 'noisy' mode the returned mask is fixed given the seed (or random if seed None).
    - For 'dynamic' mode the returned mask may change with iteration (uses iteration to reseed).
    """
    global MODE, BLOCKED_FRACTION, BLOCKED_COUNT, PENALTY, SEED
    if MODE == "standard":
        return None
    rng_seed = SEED if SEED is not None else None
    cache_key = (
        int(n),
        str(MODE),
        None if BLOCKED_COUNT is None else int(BLOCKED_COUNT),
        float(BLOCKED_FRACTION),
        None if rng_seed is None else int(rng_seed),
    )
    if MODE == "noisy" and cache_key in NOISY_MASK_CACHE:
        return NOISY_MASK_CACHE[cache_key].copy()
    if MODE == "dynamic" and iteration is not None:
        # vary with iteration to simulate changing inaccessible edges
        rng_seed = (SEED or 0) + int(iteration) * 7919
    rng = np.random.default_rng(rng_seed)
    global LAST_RELAXED_UNBLOCKED
    mask = _random_block_mask(n, rng, count=BLOCKED_COUNT, fraction=BLOCKED_FRACTION)
    if not mask.any():
        return None
    # optionally auto-relax infeasible masks
    try:
        if AUTO_RELAX:
            mask, relaxed, unblocked = ensure_feasible_mask(n, mask, auto_relax=True)
            LAST_RELAXED_UNBLOCKED = int(unblocked or 0)
            if MODE == "noisy":
                NOISY_MASK_CACHE[cache_key] = mask.copy()
            return mask
    except Exception:
        pass
    LAST_RELAXED_UNBLOCKED = 0
    if MODE == "noisy":
        NOISY_MASK_CACHE[cache_key] = mask.copy()
    return mask


def get_penalty_matrix(n: int, iteration: int | None = None) -> Optional[np.ndarray]:
    mask = get_block_mask(n, iteration=iteration)
    if mask is None:
        return None
    matrix = np.zeros((n, n), dtype=float)
    matrix[mask] = PENALTY
    return matrix


def tour_has_blocked_edge(tour, mask: Optional[np.ndarray]) -> bool:
    if mask is None:
        return False
    n = len(tour)
    for i in range(n):
        a = int(tour[i])
        b = int(tour[(i + 1) % n])
        if bool(mask[a, b]):
            return True
    return False


def edge_is_blocked(a: int, b: int, mask: Optional[np.ndarray]) -> bool:
    if mask is None:
        return False
    return bool(mask[int(a), int(b)])


def random_valid_tour(n: int, rng: np.random.Generator, mask: Optional[np.ndarray], max_attempts: int = 500):
    for _ in range(max_attempts):
        tour = rng.permutation(n).astype(int).tolist()
        if not tour_has_blocked_edge(tour, mask):
            return tour
    return None


def _mask_is_connected(mask: np.ndarray) -> bool:
    # mask is boolean symmetric: True means blocked. Build adjacency of allowed edges.
    n = mask.shape[0]
    # Allowed adjacency
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if not bool(mask[i, j]):
                adj[i].append(j)
                adj[j].append(i)
    # BFS from node 0
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n


def is_mask_feasible(mask: Optional[np.ndarray]) -> bool:
    """Approximate feasibility check for a blocked-edge mask.

    Returns True if the allowed-edge graph is connected and every node has degree >= 2.
    This is a sufficient (but not necessary) condition for the existence of a Hamiltonian cycle in practice for our demo sizes.
    """
    if mask is None:
        return True
    n = mask.shape[0]
    # compute allowed-degree
    deg = [(~mask[i]).sum() - 1 for i in range(n)]
    # deg counts self excluded; require degree >= 2
    if any(d < 2 for d in deg):
        return False
    # connectivity check
    if not _mask_is_connected(mask):
        return False
    # for small n, run a stronger Hamiltonian cycle check
    n = mask.shape[0]
    HAMILTONIAN_CHECK_THRESHOLD = 11
    if n <= HAMILTONIAN_CHECK_THRESHOLD:
        return has_hamiltonian_cycle(mask)
    return True


def has_hamiltonian_cycle(mask: np.ndarray) -> bool:
    """Brute-force/backtracking Hamiltonian cycle existence check on allowed-edge graph.

    Only used for small n (<= HAMILTONIAN_CHECK_THRESHOLD).
    """
    n = mask.shape[0]
    # build adjacency list for allowed edges
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and not bool(mask[i, j]):
                adj[i].append(j)

    # backtracking with bitmask
    target = (1 << n) - 1

    from functools import lru_cache

    @lru_cache(None)
    def dfs(node: int, visited_mask: int) -> bool:
        if visited_mask == target:
            # check if can return to start (0)
            return (0 in adj[node])
        for nb in adj[node]:
            bit = 1 << nb
            if visited_mask & bit:
                continue
            if dfs(nb, visited_mask | bit):
                return True
        return False

    # try starting at node 0
    return dfs(0, 1 << 0)


def ensure_feasible_mask(n: int, mask: Optional[np.ndarray], auto_relax: bool = False, max_relax_steps: int = 10) -> tuple[Optional[np.ndarray], bool, int]:
    """Ensure the provided mask is feasible. If `auto_relax` is True, try unblocking edges until feasible.

    Returns (mask, relaxed, unblocked_count) where `relaxed` is True if mask was modified and `unblocked_count` is how many undirected edges were unblocked.
    """
    if mask is None:
        return None, False, 0
    if is_mask_feasible(mask):
        return mask, False, 0
    if not auto_relax:
        return mask, False, 0
    # attempt to relax by unblocking a fraction of blocked edges iteratively
    relaxed = False
    cur_mask = mask.copy()
    blocked_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if bool(cur_mask[i, j])]
    original_blocked = len(blocked_edges)
    rng = np.random.default_rng(SEED)
    for step in range(max_relax_steps):
        if not blocked_edges:
            break
        # unblock a fraction of remaining blocked edges (start with 30%)
        k = max(1, int(max(1, len(blocked_edges) * 0.3 * (1.0 - float(step) / float(max_relax_steps)))))
        to_unblock = rng.choice(len(blocked_edges), size=min(k, len(blocked_edges)), replace=False)
        for idx in sorted(to_unblock, reverse=True):
            i, j = blocked_edges[idx]
            cur_mask[i, j] = False
            cur_mask[j, i] = False
            blocked_edges.pop(idx)
        if is_mask_feasible(cur_mask):
            relaxed = True
            unblocked = original_blocked - len(blocked_edges)
            return cur_mask, relaxed, unblocked
    # final attempt: try removing all blocked edges
    if is_mask_feasible(np.zeros((n, n), dtype=bool)):
        unblocked = original_blocked
        return np.zeros((n, n), dtype=bool), True, unblocked
    return mask, relaxed, 0

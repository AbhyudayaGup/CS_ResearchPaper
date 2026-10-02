from __future__ import annotations

from src import dynamic_env


def test_noisy_mode_uses_requested_blocked_edge_count():
    dynamic_env.set_params(mode="noisy", blocked_fraction=0.0, blocked_count=3, seed=17)
    try:
        mask = dynamic_env.get_block_mask(8)
        assert mask is not None
        assert int(mask.sum()) // 2 == 3
        assert (mask == mask.T).all()
    finally:
        dynamic_env.clear_params()
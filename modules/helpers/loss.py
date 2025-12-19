from typing import List, Literal

DEPTH_LOSSES = {"linear", "uniform"}


def depth_loss_weights(depth: int, depth_loss_type: str) -> List[float]:
    assert depth_loss_type in DEPTH_LOSSES, f"Invalid depth loss type: {depth_loss_type}"
    if depth_loss_type == "linear":
        return [(i+1) / (depth) for i in range(depth)]
    elif depth_loss_type == "uniform":
        return [1.0 / depth] * depth

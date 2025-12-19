import torch
import torch.nn.attention.flex_attention as flex_attn


def block_mask(q_height, kv_width):
    def _block_mask(b, h, q_idx, kv_idx):
        a = kv_idx <= kv_width
        b = q_idx >= q_height
        return a & b

    return _block_mask


def create_block_mask(token_granularities):
    token_granularities = [0, *token_granularities]
    masks = []
    for i in range(1, len(token_granularities)):
        prev_gran = token_granularities[i - 1]
        gran = token_granularities[i]
        mask = block_mask(prev_gran, gran - 1)
        masks.append(mask)
    return flex_attn.or_masks(*masks)


def create_causal_mask():
    def _causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    return _causal_mask


def create_bidir_mask(batch_widths: torch.Tensor):
    # batch_widths: (1,)
    if batch_widths.shape[0] == 1:

        def _block_mask(b, h, q_idx, kv_idx):
            a = kv_idx <= batch_widths[0]
            b = q_idx <= batch_widths[0]
            return a & b

        return _block_mask

    # batch_widths: (B,)
    def _block_mask(b, h, q_idx, kv_idx):
        a = kv_idx <= batch_widths[b]
        b = q_idx <= batch_widths[b]
        return a & b

    return _block_mask


def generate_alibi_bias(H: int) -> flex_attn._score_mod_signature:
    """Returns an alibi bias score_mod given the number of heads H

    Args:
        H: number of heads

    Returns:
        alibi_bias: alibi bias score_mod
    """

    def alibi_mod(score, b, h, q_idx, kv_idx):
        scale = torch.exp2(-((h + 1) * 8.0 / H))
        bias = torch.sqrt(torch.abs((kv_idx - q_idx))) * scale
        return score + bias

    return alibi_mod

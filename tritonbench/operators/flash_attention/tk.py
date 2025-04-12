from typing import Tuple

import thunderkittens as tk
import torch


class _TKAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal) -> torch.Tensor:
        saved_tensors = [q, k, v]
        ctx.save_for_backward(*saved_tensors)
        ctx.causal = causal
        return tk.mha_forward(q, k, v, causal)

    @staticmethod
    def backward(ctx, do) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        causal = ctx.causal
        q, k, v = ctx.saved_tensors
        o = torch.zeros_like(q).contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        l_vec = torch.zeros(
            q.shape[0],
            q.shape[1],
            q.shape[2],
            1,
            device=q.device,
            dtype=torch.float,
            requires_grad=False,
        ).contiguous()
        dq, dk, dv = tk.mha_backward(q, k, v, o, l_vec, do, causal)
        return dq, dk, dv, None


def tk_attn(q, k, v, causal) -> torch.Tensor:
    o = _TKAttn.apply(q, k, v, causal)
    return o

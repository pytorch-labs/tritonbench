import torch
import triton

from tritonbench.utils.path_utils import add_path, SUBMODULE_PATH
from tritonbench.utils.triton_op import IS_FBCODE

if IS_FBCODE:
    # Internal Import
    from hammer.ops.triton.triton_hstu_attention import triton_hstu_mha
else:
    # OSS Import
    with add_path(str(SUBMODULE_PATH.joinpath("generative-recommenders"))):
        from generative_recommenders.ops.triton import triton_hstu_mha

from typing import Tuple

class RaggedHSTUAttn(torch.nn.Module):
    def __init__(
        self,
        batch_size,
        num_heads,
        num_buckets,
        sparsity,
        target_size,
        sort_by_length,
        max_seq_len,
        causal,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.sparsity = sparsity
        self.target_size = target_size
        self.sort_by_length = sort_by_length
        self.max_seq_len = max_seq_len
        self.causal = causal

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
        num_targets: torch.Tensor,
    ) -> torch.Tensor:

        out = torch.zeros_like(v)
        # hardcode the parameters
        alpha = 0.08838834764831843
        contextual_seq_len = 0
        max_attn_len = None

        kwargs = {
            "N": self.max_seq_len,
            "alpha": alpha,
            "q": q,
            "k": k,
            "v": v,
            "seq_offsets": seq_offsets,
            "causal": self.causal,
            "num_targets": num_targets,
            "max_attn_len": max_attn_len,
            "contextual_seq_len": contextual_seq_len,
            "sort_by_length": self.sort_by_length,
        }

        out = triton_hstu_mha.apply(**kwargs)

        return out


def generate_sparse_seq_len(
    size: int,
    max_seq_len: int,
    sparsity: float,
    device: torch.device,
) -> torch.Tensor:
    if sparsity == 0.0:
        return torch.zeros(size=(size,), device=device, dtype=torch.int)
    elif sparsity == 1.0:
        return torch.ones(size=(size,), device=device, dtype=torch.int) * max_seq_len
    elif sparsity >= 0.5:
        min_seq_len: int = int((2 * sparsity - 1.0) * max_seq_len)
        return torch.randint(
            low=min_seq_len,
            high=max_seq_len,
            size=(size,),
            device=device,
            dtype=torch.int,
        )
    else:
        min_seq_len: int = 0
        max_seq_len: int = int(2 * sparsity * max_seq_len)
        return torch.randint(
            low=min_seq_len,
            high=max_seq_len,
            size=(size,),
            device=device,
            dtype=torch.int,
        )


try:
    from hammer.benchmark.module_factory.hstu_utils import (
        apply_SL,
    )
except ImportError:
    def apply_SL(lengths: torch.Tensor, alpha: float, max_seq_len: int):
        return lengths

def get_test_inputs(
    batch_size,
    num_heads,
    attn_dim,
    hidden_dim,
    max_seq_len,
    sparsity,
    target_size,
    requires_grad,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = generate_sparse_seq_len(
        size=batch_size,
        max_seq_len=max_seq_len,
        sparsity=sparsity,
        device=torch.device("cuda"),
    )
    lengths = apply_SL(lengths, alpha=2.0, max_seq_len=max_seq_len)
    # assume has_delta_q is False
    num_targets = None
    if target_size != 0:
        num_targets = torch.randint(
            1,
            target_size + 1,
            (batch_size,),
            device=lengths.device,
            dtype=lengths.dtype,
        )
        num_targets = torch.where(num_targets > lengths, lengths, num_targets)
    seq_offsets = torch.zeros(
        (batch_size + 1,),
        dtype=torch.int64,
        device="cuda",
    )
    seq_offsets[1:] = torch.cumsum(
        lengths,
        dim=0,
    )
    L = int(seq_offsets[-1].item())

    qkv = torch.randn(
        (L, num_heads, attn_dim * 2 + hidden_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    q, k, v = torch.split(qkv, [attn_dim, attn_dim, hidden_dim], dim=-1)
    q.requires_grad_(requires_grad)
    k.requires_grad_(requires_grad)
    v.requires_grad_(requires_grad)
    return q, k, v, seq_offsets, num_targets, max_seq_len

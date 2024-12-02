import torch
import triton

from tritonbench.utils.path_utils import add_path, SUBMODULE_PATH
from tritonbench.utils.triton_op import IS_FBCODE

try:
    from hammer.ops.triton.utils import prev_power_of_2

    # Internal Import
    from hammer.oss.generative_recommenders.ops.triton.triton_ragged_hstu_attention import (
        _ragged_hstu_attn_fwd_persistent,
        RaggedAttentionRelativeBiasFunction,
    )
except ModuleNotFoundError:
    # OSS Import
    with add_path(str(SUBMODULE_PATH.joinpath("generative-recommenders"))):
        from generative_recommenders.ops.triton import triton_ragged_hstu_attention

        _ragged_hstu_attn_fwd_persistent = (
            triton_ragged_hstu_attention._ragged_hstu_attn_fwd_persistent
        )
        RaggedAttentionRelativeBiasFunction = (
            triton_ragged_hstu_attention.RaggedAttentionRelativeBiasFunction
        )

    @torch.fx.wrap
    def prev_power_of_2(x: int) -> int:
        if torch.compiler.is_compiling():
            # Re-write to make Dynamo happy
            x_tensor = torch.scalar_tensor(x, dtype=torch.int64)  # type: ignore[arg-type]
            x_tensor_orig = x_tensor.clone()
            out = triton.next_power_of_2(x_tensor)  # type: ignore[arg-type]
            return int(torch.where(torch.lt(x_tensor_orig, out), out // 2, out).item())  # type: ignore[return-value]
        else:
            out = triton.next_power_of_2(x)
            return out // 2 if out > x else out


from typing import Tuple


class RaggedHSTUAttn(torch.nn.Module):
    def __init__(
        self,
        batch_size,
        num_heads,
        max_seq_len,
        num_buckets,
        sparsity,
        target_size,
        sort_by_length,
        requires_grad,
        persistent_kernel: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets
        self.sparsity = sparsity
        self.target_size = target_size
        self.sort_by_length = sort_by_length
        self.all_ts_weights = torch.nn.Parameter(
            torch.randn(
                (self.num_buckets + 1,),
                dtype=torch.float32,
            )
            .requires_grad_(requires_grad)
            .cuda()
        )
        self.all_pos_weights = torch.nn.Parameter(
            torch.randn(
                (2 * self.max_seq_len - 1,),
                dtype=torch.float32,
            )
            .requires_grad_(requires_grad)
            .cuda()
        )
        self.persistent_kernel = persistent_kernel

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
        timestamps: torch.Tensor,
        num_targets: torch.Tensor,
    ) -> torch.Tensor:
        NUM_BUCKETS = self.num_buckets
        torch._check(timestamps.size(0) + 1 == seq_offsets.size(0))

        out = torch.zeros_like(v)

        Z = timestamps.size(0)
        N = timestamps.size(1) - 1
        _, H, DimQ = q.shape
        _, _, DimV = v.shape

        kwargs = {
            "Q": q,
            "K": k,
            "V": v,
            "seq_offsets": seq_offsets,
            "delta_x_offsets": None,
            "TS": timestamps,
            "TW": self.all_ts_weights,
            "PW": self.all_pos_weights,
            "Bias": None,
            "seq2_offsets": None,
            "num_targets": num_targets,
            "Scale": None,
            "Out": out,
            "stride_qm": q.stride(0),
            "stride_qh": q.stride(1),
            "stride_kn": k.stride(0),
            "stride_kh": k.stride(1),
            "stride_vn": v.stride(0),
            "stride_vh": v.stride(1),
            "stride_sz": None,
            "stride_sm": None,
            "stride_ts": timestamps.stride(0),
            "stride_om": out.stride(0),
            "stride_oh": out.stride(1),
            "alpha": 0.08838834764831843,
            "Z": Z,
            "H": H,
            "MAX_SEQ_LEN": N,
            "AUTOTUNE_MAX_SEQ_LEN": prev_power_of_2(N),
            "DimQ": DimQ,
            "DimV": DimV,
            "DeltaSize": None,
            "num_buckets": NUM_BUCKETS,
            "max_pos_ind": None,
            "time_bucket_incr": 60,
            "time_bucket_div": 1.0,
            "time_delta": 0.0,
            "INVALID_MASK_TYPE": "lower_triangular",
            "CAUSAL": True,
            "BUCKET_FN": "sqrt",
            "ATTN_BIAS_TYPE": "ALL",
            "USE_TIME_BIAS": False,
            "USE_POS_BIAS": False,
            "HAS_MAX_POS_IND": False,
            "HAS_MULTIPLE_TARGETS": False,
            "HAS_ATTN_SCALE": False,
            "IS_DELTA_Q": False,
            "ALLOW_TF32": True,
            "BLOCK_D_Q": DimQ,
            "BLOCK_D_V": DimV,
            "MAX_ATTN_LEN": None,
            "CONTEXTUAL_SEQ_LEN": 0,
            "HAS_SORT_BY_LENGTH_INDICES": False,
            "sort_by_length_indices": None,
        }

        if self.persistent_kernel:
            grid = (1216,)
            _ragged_hstu_attn_fwd_persistent[grid](**kwargs)
        else:
            out = RaggedAttentionRelativeBiasFunction.apply(
                self.max_seq_len,  # N
                kwargs["alpha"],
                q,
                k,
                v,
                kwargs["seq_offsets"],
                kwargs["INVALID_MASK_TYPE"],
                timestamps,
                self.all_ts_weights,  # ts_weights
                self.all_pos_weights,  # pos_weights
                kwargs["CAUSAL"],  # causal,
                kwargs["num_buckets"],  # num_buckets
                "sqrt",  # time_bucket_fn
                kwargs["time_bucket_incr"],  # time_bucket_incr
                kwargs["time_bucket_div"],  # time_bucket_div
                kwargs["time_delta"],  # time_delta
                kwargs["max_pos_ind"],  # max_pos_ind
                kwargs["num_targets"],
                kwargs["ATTN_BIAS_TYPE"],  # relative_bias_type
                kwargs["MAX_ATTN_LEN"],  # max_attn_len
                kwargs["CONTEXTUAL_SEQ_LEN"],  # contextual_seq_len
                self.sort_by_length,
            )

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
        generate_hstu_timestamps,
    )
except ImportError:

    def apply_SL(lengths: torch.Tensor, alpha: float, max_seq_len: int):
        return lengths

    def generate_hstu_timestamps(batch_size, seq_len):
        ts = torch.rand(batch_size, seq_len + 1, device="cuda") ** -0.8
        ts = torch.clamp(torch.abs(ts * 86400), max=1e7)
        ts, _ = torch.sort(ts, dim=1)
        return ts.long()


def get_test_inputs(
    batch_size,
    num_heads,
    attn_dim,
    hidden_dim,
    max_seq_len,
    sparsity,
    target_size,
    sort_by_length,
    requires_grad,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    timestamps = generate_hstu_timestamps(batch_size, max_seq_len)
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
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    return q, k, v, seq_offsets, timestamps, num_targets, max_seq_len

import os
import logging
import shutil
from contextlib import contextmanager, ExitStack

from typing import Optional, List

log = logging.getLogger(__name__)

MAIN_RANDOM_SEED = 1337
AVAILABLE_PRECISIONS = [
    "bypass",
    "fp32",
    "tf32",
    "fp16",
    "amp",
    "fx_int8",
    "bf16",
    "amp_fp16",
    "amp_bf16",
    "fp8",
]

def set_random_seed():
    """Make torch manual seed deterministic. Helps with accuracy testing."""
    import random

    import numpy
    import torch

    def deterministic_torch_manual_seed(*args, **kwargs):
        from torch._C import default_generator

        seed = MAIN_RANDOM_SEED
        import torch.cuda

        if not torch.cuda._is_in_bad_fork():
            torch.cuda.manual_seed_all(seed)

        import torch.xpu

        if not torch.xpu._is_in_bad_fork():
            torch.xpu.manual_seed_all(seed)
        return default_generator.manual_seed(seed)

    torch.manual_seed(MAIN_RANDOM_SEED)
    random.seed(MAIN_RANDOM_SEED)
    numpy.random.seed(MAIN_RANDOM_SEED)
    torch.manual_seed = deterministic_torch_manual_seed

@contextmanager
def nested(*contexts):
    """
    Chain and apply a list of contexts
    """
    with ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx())
        yield contexts

@contextmanager
def fresh_inductor_cache(parallel_compile=False):
    INDUCTOR_DIR = f"/tmp/torchinductor_{os.environ['USER']}"
    if os.path.exists(INDUCTOR_DIR):
        shutil.rmtree(INDUCTOR_DIR)
    if parallel_compile:
        old_parallel_compile_threads = os.environ.get(
            "TORCHINDUCTOR_COMPILE_THREADS", None
        )
        cpu_count: Optional[int] = os.cpu_count()
        if cpu_count is not None and cpu_count > 1:
            cpu_count = min(32, cpu_count)
            log.warning(f"Set env var TORCHINDUCTOR_COMPILE_THREADS to {cpu_count}")
            os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(cpu_count)
    yield
    # clean up parallel compile directory and env
    if parallel_compile and "TORCHINDUCTOR_COMPILE_THREADS" in os.environ:
        if old_parallel_compile_threads:
            os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = old_parallel_compile_threads
        else:
            del os.environ["TORCHINDUCTOR_COMPILE_THREADS"]
    if os.path.exists(INDUCTOR_DIR):
        shutil.rmtree(INDUCTOR_DIR)


@contextmanager
def fresh_triton_cache():
    """
    Run with a fresh triton cache.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        old = os.environ.get("TRITON_CACHE_DIR", None)
        os.environ["TRITON_CACHE_DIR"] = tmpdir
        old_cache_manager = os.environ.get("TRITON_CACHE_MANAGER", None)
        os.environ.pop("TRITON_CACHE_MANAGER", None)
        yield
        if old:
            os.environ["TRITON_CACHE_DIR"] = old
        else:
            del os.environ["TRITON_CACHE_DIR"]
        if old_cache_manager:
            os.environ["TRITON_CACHE_MANAGER"] = old_cache_manager


def parse_decoration_args(
    model: "torchbenchmark.util.model.BenchmarkModel", extra_args: List[str]
) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--distributed",
        choices=["ddp", "ddp_no_static_graph", "fsdp"],
        default=None,
        help="Enable distributed trainer",
    )
    parser.add_argument(
        "--distributed_wrap_fn",
        type=str,
        default=None,
        help="Path to function that will apply distributed wrapping fn(model, dargs.distributed)",
    )
    parser.add_argument(
        "--precision",
        choices=AVAILABLE_PRECISIONS,
        default=get_precision_default(model),
        help=f"choose precisions from {AVAILABLE_PRECISIONS}",
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="enable channels-last memory layout",
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Check accuracy of the model only instead of running the performance test.",
    )
    parser.add_argument(
        "--use_cosine_similarity",
        action="store_true",
        help="use cosine similarity for correctness check",
    )
    parser.add_argument(
        "--quant-engine",
        choices=QUANT_ENGINES,
        default="x86",
        help=f"choose quantization engine for fx_int8 precision from {QUANT_ENGINES}",
    )
    parser.add_argument(
        "--num-batch",
        type=int,
        help="Number of batches if running the multi-batch train test.",
    )
    dargs, opt_args = parser.parse_known_args(extra_args)
    if not check_precision(model, dargs.precision):
        raise NotImplementedError(
            f"precision value: {dargs.precision}, "
            "amp is only supported if cuda+eval, or if `enable_amp` implemented,"
            "or if model uses staged train interfaces (forward, backward, optimizer_step)."
        )
    if not check_memory_layout(model, dargs.channels_last):
        raise NotImplementedError(
            f"Specified channels_last: {dargs.channels_last} ,"
            f" but the model doesn't implement the enable_channels_last() interface."
        )
    if not check_distributed_trainer(model, dargs.distributed):
        raise NotImplementedError(
            f"We only support distributed trainer {dargs.distributed} for train tests, "
            f"but get test: {model.test}"
        )
    return (dargs, opt_args)


def apply_precision(
    op,
    precision: str,
):
    if precision == "bypass" or precision == "fp32":
        return
    if precision == "fp16":
        op.enable_fp16()
    elif precision == "bf16":
        op.enable_bf16()
    elif precision == "tf32":
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        log.warning(f"[tritonbench] Precision {precision} is handled by operator.")

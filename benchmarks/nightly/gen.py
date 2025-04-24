# Generate the nightly benchmark config to autogen.yaml
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent

METADATA_PATH = REPO_PATH.joinpath("benchmarks/gen_metadata/metadata")
OUTPUT_PATH = Path(os.path.abspath(__file__)).parent.joinpath("autogen.yaml")


def get_metadata(name: str) -> Any:
    fpath = os.path.join(METADATA_PATH, f"{name}.yaml")
    with open(fpath, "r") as f:
        return yaml.safe_load(f)


TRITON_OPS = get_metadata("triton_operators")
DTYPE_OPS = get_metadata("dtype_operators")
TFLOPS_OPS = get_metadata("tflops_operators")
BASELINE_OPS = get_metadata("baseline_operators")
BWD_OPS = get_metadata("backward_operators")


def gen_run(operators: List[str], bwd: bool = False) -> Dict[str, Any]:
    out = {}
    for op in operators:
        dtype = (
            DTYPE_OPS[op]
            if not DTYPE_OPS[op] == "fp8" and not DTYPE_OPS[op] == "bypass"
            else ""
        )
        mode = "fwd" if not bwd else "bwd"
        run_name = f"{dtype}_{op}_{mode}" if dtype else f"{op}_{mode}"
        cmd = ["--op", op]
        # add metrics
        metrics = []
        if op in TFLOPS_OPS:
            metrics.append("tflops")
        if op in BASELINE_OPS and not (
            BASELINE_OPS[op] in TRITON_OPS[op] and len(TRITON_OPS[op]) == 1
        ):
            cmd.extend(["--baseline", BASELINE_OPS[op]])
            metrics.append("speedup")
        cmd.extend(["--metrics", ",".join(metrics)])
        # add mode
        if bwd:
            cmd.append("--bwd")
        # add backends
        run_backends = TRITON_OPS[op]
        if op in BASELINE_OPS and not BASELINE_OPS[op] in run_backends:
            run_backends.append(BASELINE_OPS[op])
        cmd.extend(["--only", ",".join(run_backends)])
        out[run_name] = {}
        out[run_name]["op"] = op
        out[run_name]["args"] = " ".join(cmd)
    return out


def run():
    # generate forward runs
    forward_ops = [op for op in TRITON_OPS if op in TFLOPS_OPS]
    runs = gen_run(forward_ops)
    # generate backward runs
    backward_ops = [op for op in BWD_OPS if op in TRITON_OPS and op in TFLOPS_OPS]
    runs.update(gen_run(backward_ops, bwd=True))
    with open(OUTPUT_PATH, "w") as f:
        yaml.safe_dump(runs, f, sort_keys=False)


if __name__ == "__main__":
    run()

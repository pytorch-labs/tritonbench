import logging
import os
import subprocess
from contextlib import contextmanager
from typing import Dict

# NVIDIA A100 GPU Spec:
# https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
NV_A100 = {
    "fp32": 19.5,
    "tf32": 156,
    "bf16": 312,
    "fp16": 312,
}

# NVIDIA H100 GPU Datasheet:
# https://www.nvidia.com/en-gb/data-center/h100
NV_H100 = {
    "fp32": 51,
    "tf32": 756,
    "bf16": 1513,
    "fp16": 1513,
    "fp8": 3026,
}


HW_ROOFLINE_SPECS: Dict[
    bool, Dict[str, Dict[str, float]]
] = {  # true is compute bound false would be memory bound
    True: {
        "NVIDIA A100-SXM4-40GB": NV_A100,
        "NVIDIA A100-PG509-200": NV_A100,
        "NVIDIA H100": NV_H100,
    },
    False: {
        # https://www.nvidia.com/en-gb/data-center/h100
        # values in gbps
        "NVIDIA H100": 2000,
    },
}

CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

POWER_LIMIT = {
    "NVIDIA PG509-210": "330",
    "NVIDIA A100": "330",
    "NVIDIA H100": "650",
}
FREQ_LIMIT = {
    "NVIDIA PG509-210": "1410",
    "NVIDIA A100": "1410",
    "NVIDIA H100": "1980",
}


def _set_pm():
    command = ["sudo", "nvidia-smi", "-i", CUDA_VISIBLE_DEVICES, "-pm", "1"]
    subprocess.check_call(command)


def _set_power(gpu_info: str):
    command = [
        "sudo",
        "nvidia-smi",
        "-i",
        CUDA_VISIBLE_DEVICES,
        "--power-limit",
        POWER_LIMIT[gpu_info],
    ]
    subprocess.check_call(command)


def _set_clock(gpu_info: str):
    # lgc: lock gpu clocks
    command = [
        "sudo",
        "nvidia-smi",
        "-i",
        CUDA_VISIBLE_DEVICES,
        "-lgc",
        FREQ_LIMIT[gpu_info],
    ]
    subprocess.check_call(command)


def _reset_clock(gpu_info: str):
    # rgc: reset gpu clocks
    command = ["sudo", "nvidia-smi", "-i", CUDA_VISIBLE_DEVICES, "-rgc"]
    subprocess.check_call(command)


def _get_gpu_name() -> str:
    import pynvml  # @manual=fbsource//third-party/pypi/nvidia-ml-py:nvidia-ml-py

    pynvml.nvmlInit()
    gpu_id = CUDA_VISIBLE_DEVICES.split(",")[0]
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_id))
    return pynvml.nvmlDeviceGetName(handle).decode("utf-8")


@contextmanager
def gpu_lockdown(enabled=True):
    try:
        if enabled:
            logging.info(f"[tritonbench] Locking down GPU {CUDA_VISIBLE_DEVICES}")
            gpu_name = _get_gpu_name()
            assert gpu_name in POWER_LIMIT, f"Unsupported GPU {gpu_name}"
            _set_pm()
            _set_power(gpu_name)
            _set_clock(gpu_name)
        yield
    finally:
        if enabled:
            _reset_clock(gpu_name)

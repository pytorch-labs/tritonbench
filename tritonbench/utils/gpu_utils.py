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
# "https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
NV_H100 = {
    "fp32": 989 // 2,
    "tf32": 989 // 2,
    "bf16": 1979 // 2,
    "fp16": 1979 // 2,
    "fp8": 3958 // 2,
    "int8": 3958 // 2,
}

# https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-platform-data-sheet.pdf
AMD_MI300X = {
    "fp32": 1300 // 8,
    "tf32": 5200 // 8,
    "bf16": 10500 // 8,
    "fp16": 10500 // 8,
    "fp8": 20900 // 8,
    "int8": 20900 // 8,
}


HW_ROOFLINE_SPECS: Dict[
    bool, Dict[str, Dict[str, float]]
] = {  # true is compute bound false would be memory bound
    True: {
        "NVIDIA A100-SXM4-40GB": NV_A100,
        "NVIDIA A100-PG509-200": NV_A100,
        "NVIDIA H100": NV_H100,
        "AMD MI300X": AMD_MI300X,
    },
    False: {
        # https://www.nvidia.com/en-gb/data-center/h100
        # values in gbps
        "NVIDIA H100": 2000,
        # https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-platform-data-sheet.pdf
        "AMD MI300X": 5300,
    },
}

CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

POWER_LIMIT = {
    "NVIDIA PG509-210": "330",
    "NVIDIA A100": "330",
    "NVIDIA H100": "650",
}
GRAPHIC_FREQ_LIMIT = {
    "NVIDIA PG509-210": "1410",
    "NVIDIA A100": "1410",
    "NVIDIA H100": "1980",
}
MEMORY_FREQ_LIMIT = {
    "NVIDIA H100": "1593",
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
        GRAPHIC_FREQ_LIMIT[gpu_info],
    ]
    subprocess.check_call(command)
    # lmc: lock memory clocks
    if gpu_info in MEMORY_FREQ_LIMIT:
        command = [
            "sudo",
            "nvidia-smi",
            "-i",
            CUDA_VISIBLE_DEVICES,
            "-lmc",
            MEMORY_FREQ_LIMIT[gpu_info],
        ]
        subprocess.check_call(command)


def _maybe_set_app_clocks(gpu_info: str):
    graphic_freq = GRAPHIC_FREQ_LIMIT.get(gpu_info, None)
    memory_freq = MEMORY_FREQ_LIMIT.get(gpu_info, None)
    if graphic_freq and memory_freq:
        command = [
            "sudo",
            "nvidia-smi",
            "-i",
            CUDA_VISIBLE_DEVICES,
            "-ac",
            f"{memory_freq},{graphic_freq}",
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
    return pynvml.nvmlDeviceGetName(handle)


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
            _maybe_set_app_clocks(gpu_name)
        yield
    finally:
        if enabled:
            gpu_name = _get_gpu_name()
            _reset_clock(gpu_name)

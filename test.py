"""test.py
Setup and test operators.
"""
import torch
import unittest

from test import load_cpu_tests, load_gpu_tests


TEST_MAP = {
    "cpu": load_cpu_tests,
    "gpu": load_gpu_tests,
}

def _load_tests():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for device in devices:
        load_test = TEST_MAP[device]
        load_test()

_load_tests()

if __name__ == "__main__":
    unittest.main()
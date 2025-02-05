"""
Convert Tritonbench json to ClickHouse oss_ci_benchmark_v3 schema.
https://github.com/pytorch/test-infra/blob/main/clickhouse_db_schema/oss_ci_benchmark_v3/schema.sql
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path

from typing import Dict, Any, List

def parse_dependencies():
    pass

def maybe_get_target_value(metrics: Dict[str, float], metric_id: str) -> float:
    target_metric_id = f"{metric_id}-target"
    return metrics.get(target_metric_id, 0.0)

def generate_oss_ci_benchmark_v3_json(benchmark_result: Dict[str, Any]) -> List[List[Any]]:
    """
    Parse Benchmark Json and return a list of entries
    """
    common = {}
    common["timestamp"] = benchmark_result["env"]["benchmark_date"]
    common["name"] = benchmark_result["name"]
    common["repo"] = benchmark_result["github"]["GITHUB_REPOSITORY"]
    common["head_branch"] = benchmark_result["github"]["GITHUB_REF"]
    common["head_sha"] = benchmark_result["github"]["GITHUB_WORKFLOW_SHA"]
    common["workflow_id"] = benchmark_result["github"]["GITHUB_WORKFLOW"]
    common["run_attempt"] = benchmark_result["github"]["GITHUB_RUN_ATTEMPT"]
    common["job_id"] = benchmark_result["github"]["GITHUB_JOB"]
    common["runners"] = [
        {
            "name": benchmark_result["github"]["RUNNER_NAME"],
            "type": benchmark_result["github"]["RUNNER_TYPE"],
            "cpu_info": "not_available",
            "cpu_count": 0,
            "mem_info": "not_available",
            "avail_mem_in_gb": 0,
            "gpu_info": benchmark_result["env"]["device"],
            "gpu_count": 1,
            "gpu_mem_info": "not_available",
            "avail_gpu_mem_in_gb": 0,
            "extra_info": {},
        }
    ]
    out = []
    for metric_id in benchmark_result["metrics"]:
        # bypass if the metric is target
        if metric_id.endswith("-target"):
            continue
        entry = common.copy()
        entry["dependencies"] = parse_dependencies(benchmark_result["env"])
        mode, dtype, inputs, metric_name = parse_metric_id(metric_id)
        metric_value = benchmark_result["metrics"][metric_id]
        entry["benchmark"] = {
            "name": benchmark_result["name"],
            "mode": benchmark_result["mode"],
            "dtype": benchmark_result["dtype"],
            "extra_info": {
                "cuda_version": benchmark_result["env"]["cuda_version"],
                "conda_env": benchmark_result["env"]["conda_env"],
            },
        }
        # We use the model field for operator
        entry["model"] = {
            "name": "aaa",
            "type": "",
            "backend": "",
        }
        entry["metric"] = {
            "name": metric_name,
            "benchmark_values": [metric_value],
            "target_value": maybe_get_target_value(benchmark_result["metrics"], metric_id),
            "extra_info": {},
        }
        out.append(entry)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        required=True,
        help="Upload benchmark result json file.",
    )
    args = parser.parse_args()
    upload_file_path = Path(args.json)
    assert (
        upload_file_path.exists()
    ), f"Specified result json path {args.json} does not exist."
    with open(upload_file_path, "r") as fp:
        benchmark_result = json.load(fp)

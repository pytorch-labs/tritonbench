"""
Upload result json file to scribe.
"""

import argparse
import json
import os
import time

from collections import defaultdict

import requests

CATEGORY_NAME = "perfpipe_pytorch_user_benchmarks"
BENCHMARK_SCHEMA = {
    "int": ["time"],
    "normal": [
        "benchmark_date",
        "unix_user",
        "submission_group_id",
        "cuda_version",
        "device",
        "conda_env",
        "pytorch_commit",
        "triton_commit",
        "tritonbench_commit",
        "triton_branch",
        "pytorch_branch",
        "tritonbench_branch",
        "triton_commit_time",
        "pytorch_commit_time",
        "tritonbench_commit_time",
        "github_action",
        "github_actor",
        "github_base_ref",
        "github_ref",
        "github_ref_protected",
        "github_repository",
        "github_run_attempt",
        "github_run_id",
        "github_run_number",
        "github_workflow",
        "github_workflow_ref",
        "github_workflow_sha",
        "job_name",
        "runner_arch",
        "runner_name",
        "runner_type",
        "runner_os",
        "metric_id",
    ],
    "float": ["metric_value"],
}


class ScribeUploader:
    def __init__(self, category, schema):
        self.category = category
        self.schema = schema

    def _format_message(self, field_dict):
        assert "time" in field_dict, "Missing required Scribe field 'time'"
        message = defaultdict(dict)
        for field, value in field_dict.items():
            field = field.lower()
            if value is None:
                continue
            if field in self.schema["normal"]:
                message["normal"][field] = str(value)
            elif field in self.schema["int"]:
                message["int"][field] = int(value)
            elif field in self.schema["float"]:
                message["float"][field] = float(value)
            else:
                raise ValueError(
                    "Field {} is not currently used, "
                    "be intentional about adding new fields to schema".format(field)
                )
        return message

    def _upload(self, messages: list):
        access_token = os.environ.get("TRITONBENCH_SCRIBE_GRAPHQL_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("Can't find access token from environment variable")
        url = "https://graph.facebook.com/scribe_logs"
        r = requests.post(
            url,
            data={
                "access_token": access_token,
                "logs": json.dumps(
                    [
                        {
                            "category": self.category,
                            "message": json.dumps(message),
                            "line_escape": False,
                        }
                        for message in messages
                    ]
                ),
            },
        )
        print(r.text)
        r.raise_for_status()

    def post_benchmark_results(self, bm_data):
        messages = []
        base_message = {
            "time": int(time.time()),
        }
        base_message.update(bm_data["env"])
        base_message.update(bm_data["github"])
        base_message["submission_group_id"] = f"tritonbench.{bm_data['name']}"
        base_message["unix_user"] = "tritonbench_ci"
        for metric in bm_data["metrics"]:
            msg = base_message.copy()
            msg["metric_id"] = metric
            msg["metric_value"] = bm_data["metrics"][metric]
            formatted_msg = self._format_message(msg)
            messages.append(formatted_msg)
        self._upload(messages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json", required=True, type=argparse.FileType("r"), help="Userbenchmark json"
    )
    args = parser.parse_args()
    uploader = ScribeUploader(category=CATEGORY_NAME, schema=BENCHMARK_SCHEMA)
    benchmark_data = json.load(args.json)
    uploader.post_benchmark_results(benchmark_data)

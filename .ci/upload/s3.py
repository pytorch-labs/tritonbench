import argparse
import boto3
import json
import os
import yaml

from pathlib import Path

from typing import List, Any, Optional


TRITONBENCH_S3_BUCKET = "ossci-metrics"
TRITONBENCH_S3_OBJECT = "tritonbench"

class S3Client:
    def __init__(self, bucket, object):
        self.s3 = boto3.client("s3")
        self.bucket = bucket
        self.object = object

    def download_file(self, key: str, dest_dir: str) -> None:
        filename = S3Client.get_filename_from_key(key)
        assert filename, f"Expected non-empty filename from key {key}."
        with open(os.path.join(dest_dir, filename), "wb") as f:
            self.s3.download_fileobj(self.bucket, key, f)

    def upload_file(self, prefix: str, file_path: Path) -> None:
        file_name = file_path.name
        s3_key = (
            f"{self.object}/{prefix}/{file_name}"
            if prefix
            else f"{self.object}/{file_name}"
        )
        response = self.s3.upload_file(str(file_path), self.bucket, s3_key)
        print(f"S3 client response: {response}")

    def get_file_as_json(self, key: str) -> Any:
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))

    def get_file_as_yaml(self, key: str) -> Any:
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return yaml.safe_load(obj["Body"].read().decode("utf-8"))

    def exists(self, prefix: str, file_name: str) -> Optional[str]:
        """Test if the key object/prefix/file_name exists in the S3 bucket.
        If True, return the S3 object key. Return None otherwise."""
        s3_key = (
            f"{self.object}/{prefix}/{file_name}"
            if prefix
            else f"{self.object}/{file_name}"
        )
        result = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=s3_key)
        if "Contents" in result:
            return s3_key
        return None

    def list_directory(self, directory=None) -> List[str]:
        """List the directory files in the S3 bucket path.
        If the directory doesn't exist, report an error."""
        prefix = f"{self.object}/{directory}/" if directory else f"{self.object}/"
        pages = self.s3.get_paginator("list_objects").paginate(
            Bucket=self.bucket, Prefix=prefix
        )
        keys = filter(
            lambda x: not x == prefix, [e["Key"] for p in pages for e in p["Contents"]]
        )
        return list(keys)

    def get_filename_from_key(object_key: str) -> str:
        filename = object_key.split("/")[-1]
        return filename


def upload_s3(benchmark_name: str, runner: str, date_str: str, file_path: Path):
    """S3 path:
    s3://ossci-metrics/tritonbench/<benchmark-name>/<runner>/<date>/metrics-<YYmmddHHMMSS>.json
    """
    s3client = S3Client(TRITONBENCH_S3_BUCKET, TRITONBENCH_S3_OBJECT)
    prefix = f"{benchmark_name}/{runner}/{date_str}"
    s3client.upload_file(prefix=prefix, file_path=file_path)


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
    runner = os.environ.get("GITHUB_RUNNER", None)
    if not runner:
        runner = benchmark_result["github"]["GITHUB_RUNNER"]
    assert runner, "Expected runner name exists, get None. Pass --runner or set env GITHUB_RUNNER."
    benchmark_name = benchmark_result["name"]
    date_str = benchmark_result["benchmark_date"]
    upload_s3(benchmark_name, runner, date_str, upload_file_path)

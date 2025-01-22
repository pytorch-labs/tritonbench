"""
Upload result json file to scribe.
"""

import argparse

CATEGORY_NAME = "perfpipe_pytorch_user_benchmarks"


class ScribeUploader:
    def __init__(self, category):
        self.category = category


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json", required=True, type=argparse.FileType("r"), help="Userbenchmark json"
    )
    args = parser.parse_args()

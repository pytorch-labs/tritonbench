import time
from datetime import datetime
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent
BENCHMARKS_OUTPUT_DIR = REPO_DIR.joinpath(".benchmarks")

def setup_output_dir(bm_name: str):
    current_timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
    output_dir = BENCHMARKS_OUTPUT_DIR.joinpath(bm_name, f"run-{current_timestamp}")
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    return output_dir.absolute()

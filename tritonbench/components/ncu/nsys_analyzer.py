import os
import shutil
import sys
from typing import Dict, List
import csv

nsys_metrics = {"nsys_gpu_kernel_exec_summary"}


def import_nsys_python_path() -> None:
    """
    This function modifies the Python path to include the NVIDIA Nsight Systems (nsys) Python modules.
    It searches for the 'nsys' command in the system PATH, determines its location, and appends the
    host-specific Python library and reports directories to the Python path.

    Raises:
        FileNotFoundError: If the 'nsys' command is not found in the system PATH.
        FileNotFoundError: If the target-specific directory cannot be found.
        FileNotFoundError: If the Python library directory does not exist.
        FileNotFoundError: If the reports directory does not exist.
        ImportError: If the nsysstats module cannot be imported after path modification.
    """
    nsys_path = shutil.which("nsys")
    if not nsys_path:
        raise FileNotFoundError("Could not find 'nsys' command in PATH.")
    nsys_path = os.path.dirname(os.path.dirname(nsys_path))

    # Find target-* directory
    target_dir = None
    for item in os.listdir(nsys_path):
        if item.startswith("target-"):
            target_dir = item
            break

    if not target_dir:
        raise FileNotFoundError(f"Could not find 'target-*' directory in {nsys_path}")

    # Add host-*/python/lib to PYTHONPATH
    target_python_lib = os.path.join(nsys_path, target_dir, "python/lib")
    if not os.path.exists(target_python_lib):
        raise FileNotFoundError(
            f"Python lib directory does not exist: {target_python_lib}"
        )
    target_python_reports = os.path.join(nsys_path, target_dir, "reports")
    if not os.path.exists(target_python_reports):
        raise FileNotFoundError(
            f"Reports directory does not exist: {target_python_reports}"
        )
    sys.path.append(target_python_reports)
    sys.path.append(target_python_lib)
    try:
        import nsysstats
    except ImportError as e:
        raise ImportError(f"Failed to import nsysstats: {e}") from e


def export_to_sqlite(nsys_rep_path: str) -> str:
    """
    Export nsys report to sqlite database.

    Args:
        nsys_rep_path (str): Path to the nsys report file, e.g. /path/to/report.nsys-rep

    Returns:
        str: Path to the generated sqlite file
    """
    base_path = os.path.splitext(nsys_rep_path)[0]
    sqlite_path = base_path + ".sqlite"
    cmd = f"nsys stats --force-export=true {nsys_rep_path} --sqlite {sqlite_path} > /dev/null 2>&1"
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Failed to export nsys report to sqlite. Command: {cmd}")
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(
            f"Expected sqlite file was not created at: {sqlite_path}"
        )

    return sqlite_path


def get_kernel_exec_summary(sqlite_path: str) -> Dict[str, float]:
    from cuda_kern_exec_trace import CudaKernExecTrace

    report, exitval, errmsg = CudaKernExecTrace.Report(sqlite_path, [])
    if report is None:
        raise RuntimeError(
            f"Failed to read nsys report from {sqlite_path}. Exit code: {exitval}. Error: {errmsg}"
        )

    # Store the results
    results = {}
    headers = None

    # Define column prefixes to search for
    column_prefixes = {"name": "Kernel Name", "duration": "Kernel Dur"}
    target_columns = {key: None for key in column_prefixes}

    # Initialize results containers
    durations = []
    kernel_names = []
    duration_sum = 0
    num_of_kernels = 0

    while True:
        row = report.get_query_row()
        if row is None:
            break

        if headers is None:
            # Process headers and find column indices
            headers = row
            print("DEBUG:================")
            print(headers)
            print("DEBUG:================")
            # Find all target columns using startswith
            for key, prefix in column_prefixes.items():
                matching_cols = [
                    i for i, h in enumerate(headers) if h.startswith(prefix)
                ]
                if not matching_cols:
                    raise ValueError(
                        f"No column starting with '{prefix}' found in headers"
                    )
                if len(matching_cols) > 1:
                    raise ValueError(
                        f"Multiple columns starting with '{prefix}' found in headers"
                    )
                target_columns[key] = matching_cols[0]
            continue

        # Extract kernel name and duration from the row using target column indices
        kernel_name = row[target_columns["name"]]
        duration = float(row[target_columns["duration"]])

        kernel_names.append(kernel_name)
        durations.append(duration)
        duration_sum += duration
        num_of_kernels += 1

    results = {
        "durations": durations,
        "kernel_names": kernel_names,
        "duration_sum": duration_sum,
        "num_of_kernels": num_of_kernels,
    }
    print(results)
    return results


def read_nsys_report(
    report_path: str, required_metrics: List[str]
) -> Dict[str, List[float]]:
    assert os.path.exists(
        report_path
    ), f"The nsys report at {report_path} does not exist. Ensure you add --metrics nsys_rep to your benchmark run."
    import_nsys_python_path()
    sqlite_path = export_to_sqlite(report_path)
    # return get_kernel_exec_summary(sqlite_path)

import importlib.util

from tritonbench.utils.env_utils import is_fbcode


def tritonparse_init(tritonparse_log_path):
    if tritonparse_log_path is not None:
        # capture errors but don't fail the entire script
        try:
            if importlib.util.find_spec("tritonparse") is None:
                print(
                    "Warning: tritonparse is not installed. Run 'python install.py --tritonparse' to install it."
                )
                return
            import tritonparse.structured_logging

            tritonparse.structured_logging.init(
                tritonparse_log_path, enable_trace_launch=True
            )
            print(
                f"TritonParse structured logging initialized with log path: {tritonparse_log_path}"
            )
        except Exception as e:
            print(f"Warning: Failed to initialize tritonparse: {e}")


def tritonparse_parse(tritonparse_log_path):
    if tritonparse_log_path is not None:
        # capture errors but don't fail the entire script
        try:
            from tritonparse.utils import unified_parse

            unified_parse(tritonparse_log_path)
        except Exception as e:
            print(f"Warning: Failed to parse tritonparse log: {e}")

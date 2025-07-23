"""
Utilities for listing available metrics in tritonbench.
"""

import sys
from dataclasses import fields
from typing import Dict, List, Set

from tritonbench.operators import load_opbench_by_name
from tritonbench.operators_collection import list_operators_by_collection

from tritonbench.utils.triton_op import (
    BenchmarkOperatorMetrics,
    OVERRIDDEN_METRICS,
    REGISTERED_METRICS,
)


def get_builtin_metrics() -> List[str]:
    """Get all built-in metrics from BenchmarkOperatorMetrics dataclass."""
    return [
        field.name
        for field in fields(BenchmarkOperatorMetrics)
        if field.name != "extra_metrics"
    ]


def load_operators_to_register_metrics(operators: List[str]) -> None:
    """Load operators to trigger metrics registration."""
    for op_name in operators:
        try:
            load_opbench_by_name(op_name)
        except Exception as e:
            print(f"Warning: Failed to load operator '{op_name}': {e}", file=sys.stderr)


def get_custom_metrics_for_operators(operators: List[str]) -> Dict[str, List[str]]:
    """Get custom metrics for specific operators."""
    # Load operators to ensure their metrics are registered
    load_operators_to_register_metrics(operators)

    result = {}
    for op_name in operators:
        result[op_name] = REGISTERED_METRICS.get(op_name, [])
    return result


def get_overridden_metrics_for_operators(operators: List[str]) -> Dict[str, List[str]]:
    """Get overridden metrics for specific operators."""
    # Load operators to ensure their metrics are registered
    load_operators_to_register_metrics(operators)

    result = {}
    for op_name in operators:
        result[op_name] = OVERRIDDEN_METRICS.get(op_name, [])
    return result


def get_all_metrics_for_collection(
    collection_name: str,
) -> Dict[str, Dict[str, List[str]]]:
    """Get all metrics for operators in a collection."""
    operators = list_operators_by_collection(collection_name)
    load_operators_to_register_metrics(operators)

    result = {}
    for op_name in operators:
        result[op_name] = {
            "custom": REGISTERED_METRICS.get(op_name, []),
            "overridden": OVERRIDDEN_METRICS.get(op_name, []),
        }
    return result


def format_operator_specific_metrics(
    operators: List[str],
    builtin_metrics: List[str],
    custom_metrics: Dict[str, List[str]],
    overridden_metrics: Dict[str, List[str]],
) -> str:
    """Format metrics output for specific operators."""
    output = []

    # Show built-in metrics (common to all operators)
    output.append("Built-in metrics (available for all operators):")
    for metric in sorted(builtin_metrics):
        output.append(f"  {metric}")

    # Show metrics for each operator
    for op_name in sorted(operators):
        custom = custom_metrics.get(op_name, [])
        overridden = overridden_metrics.get(op_name, [])

        if not custom and not overridden:
            continue

        output.append(f"\nOperator: {op_name}")

        if custom:
            output.append("  Custom metrics:")
            for metric in sorted(custom):
                output.append(f"    {metric}")

        if overridden:
            output.append("  Overridden metrics:")
            for metric in sorted(overridden):
                output.append(f"    {metric}")

    return "\n".join(output)


def list_metrics(operators: List[str] = None) -> str:
    """
    List available metrics based on the provided operators.

    Args:
        operators: List of specific operators to show metrics for

    Returns:
        Formatted string with metrics information
    """
    builtin_metrics = get_builtin_metrics()

    if operators:
        # Specific operators case
        custom_metrics = get_custom_metrics_for_operators(operators)
        overridden_metrics = get_overridden_metrics_for_operators(operators)
        return format_operator_specific_metrics(
            operators, builtin_metrics, custom_metrics, overridden_metrics
        )
    else:
        # Global case - show built-in metrics only
        output = []
        output.append("Built-in metrics (available for all operators):")
        for metric in sorted(builtin_metrics):
            output.append(f"  {metric}")
        return "\n".join(output)

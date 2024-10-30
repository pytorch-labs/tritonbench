import unittest

from tritonbench.utils.parser import get_parser
from tritonbench.operators import load_opbench_by_name
from tritonbench.operators_collection import list_operators_by_collection


class TestTritonbenchCpu(unittest.TestCase):

    def _get_test_op(self):
        parser = get_parser(["--device", "cpu", "--op", "test_op"])
        tb_args, extra_args = parser.parse_known_args(
            ["--device", "cpu", "--op", "test_op"]
        )
        Operator = load_opbench_by_name(tb_args.op)
        test_op = Operator(tb_args, extra_args)
        return test_op

    def test_metric_x_only_true(
        self,
    ):  # test x_only = True argument in register_metric()
        test_op = self._get_test_op()
        test_op.run()
        benchmark_operator_result = test_op.output
        headers, table = benchmark_operator_result._table()

        self.assertIn("test_metric", headers)  # x_only = True
        self.assertNotIn(
            "test_op-test_metric", headers
        )  # test_op-test_metric occurs only when x_only = False

    def test_metric_custom_label(self):
        test_op = self._get_test_op()
        test_op.run()
        benchmark_operator_result = test_op.output
        headers, table = benchmark_operator_result._table()

        self.assertTrue(
            ["new_op_label-" in header for header in headers]
        )  # custom benchmark label should be used in headers
        self.assertFalse(
            any(["test_op-" in header for header in headers])
        )  # default benchmark label should not be present in headers

    def test_list_operators_by_collection(self):
        all_ops = list_operators_by_collection()
        self.assertTrue(len(all_ops) > 0)
        default_ops = list_operators_by_collection("default")
        self.assertTrue(len(default_ops) > 0)
        liger_ops = list_operators_by_collection("liger")
        self.assertTrue(len(liger_ops) > 0)
        self.assertTrue(liger_ops[0] not in default_ops)

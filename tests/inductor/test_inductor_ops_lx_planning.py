import functools
import copy
import dataclasses
import torch_spyre
import os
import sys
import inspect
import torch
from torch.utils import _pytree as pytree

from torch._dynamo.testing import make_test_cls_with_patches

import unittest
import utils_inductor

_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(_test_dir)

import inductor.test_inductor_ops  # noqa: E402


@dataclasses.dataclass
class TestFailure:
    suffixes: tuple[str, ...]
    is_skip: bool = False
    __test__: bool = False


def copy_tests(my_cls, other_cls, suffix, test_failures=None, xfail_prop=None):
    for name, value in my_cls.__dict__.items():
        if name.startswith("test_"):
            if "compare" not in inspect.getsource(value):
                continue

            # You cannot copy functions in Python, so we use closures here to
            # create objects with different ids. Otherwise, unittest.skip
            # would modify all methods sharing the same object id. Also, by
            # using a default argument, we create a copy instead of a
            # reference. Otherwise, we would lose access to the value.

            @functools.wraps(value)
            def new_test(self, value=value):
                return value(self)

            # Copy __dict__ which may contain test metadata
            new_test.__dict__ = copy.deepcopy(value.__dict__)

            if xfail_prop is not None and hasattr(value, xfail_prop):
                new_test = unittest.expectedFailure(new_test)

            tf = test_failures and test_failures.get(name)
            print("name", name, tf)
            if tf and suffix in tf.suffixes:
                skip_func = (
                    unittest.skip("Skipped!")
                    if tf.is_skip
                    else unittest.expectedFailure
                )
                new_test = skip_func(new_test)

            setattr(other_cls, f"{name}_{suffix}", new_test)

    # Special case convenience routine
    if hasattr(my_cls, "is_dtype_supported"):
        other_cls.is_dtype_supported = my_cls.is_dtype_supported


# FAILED tests/inductor/test_inductor_ops_lx_planning.py::LxPlanningTest::test_cat_1d_dim0_lx_planning - torch._inductor.exc.InductorError: AttributeError: 'NoneType' object has no attribute 'name'
# FAILED tests/inductor/test_inductor_ops_lx_planning.py::LxPlanningTest::test_cat_1d_dim0_three_tensors_lx_planning - torch._inductor.exc.InductorError: AttributeError: 'NoneType' object has no attribute 'name'
# FAILED tests/inductor/test_inductor_ops_lx_planning.py::LxPlanningTest::test_cat_2d_dim0_diff_size_lx_planning - torch._inductor.exc.InductorError: AttributeError: 'NoneType' object has no attribute 'name'
# FAILED tests/inductor/test_inductor_ops_lx_planning.py::LxPlanningTest::test_cat_2d_dim0_three_tensors_lx_planning - torch._inductor.exc.InductorError: AttributeError: 'NoneType' object has no attribute 'name'
# FAILED tests/inductor/test_inductor_ops_lx_planning.py::LxPlanningTest::test_cat_2d_dim1_diff_size_lx_planning - torch._inductor.exc.InductorError: AttributeError: 'NoneType' object has no attribute 'name'
# FAILED tests/inductor/test_inductor_ops_lx_planning.py::LxPlanningTest::test_cat_3d_dim0_lx_planning - torch._inductor.exc.InductorError: AttributeError: 'NoneType' object has no attribute 'name'
# FAILED tests/inductor/test_inductor_ops_lx_planning.py::LxPlanningTest::test_cat_3d_dim1_lx_planning - torch._inductor.exc.InductorError: AttributeError: 'NoneType' object has no attribute 'name'
# FAILED tests/inductor/test_inductor_ops_lx_planning.py::LxPlanningTest::test_cat_3d_dim1_size1_lx_planning - torch._inductor.exc.InductorError: AttributeError: 'NoneType' object has no attribute 'name'
# FAILED tests/inductor/test_inductor_ops_lx_planning.py::LxPlanningTest::test_cat_3d_dim2_lx_planning - torch._inductor.exc.InductorError: AttributeError: 'NoneType' object has no attribute 'name'
# FAILED tests/inductor/test_inductor_ops_lx_planning.py::LxPlanningTest::test_cat_4d_dim0_lx_planning - torch._inductor.exc.InductorError: AttributeError: 'NoneType' object has no attribute 'name'
# FAILED tests/inductor/test_inductor_ops_lx_planning.py::LxPlanningTest::test_cat_4d_dim1_lx_planning - torch._inductor.exc.InductorError: AttributeError: 'NoneType' object has no attribute 'name'
# FAILED tests/inductor/test_inductor_ops_lx_planning.py::LxPlanningTest::test_cat_4d_dim2_lx_planning - torch._inductor.exc.InductorError: AttributeError: 'NoneType' object has no attribute 'name'
# FAILED tests/inductor/test_inductor_ops_lx_planning.py::LxPlanningTest::test_cat_4d_dim3_lx_planning - torch._inductor.exc.InductorError: AttributeError: 'NoneType' object has no attribute 'name'

# xfail by default, set is_skip=True to skip
test_failures = {
    "test_cat_1d_dim0": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_1d_dim0_three_tensors": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_2d_dim0_diff_size": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_2d_dim0_three_tensors": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_2d_dim1_diff_size": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_3d_dim0": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_3d_dim1": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_3d_dim1_size1": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_3d_dim2": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_4d_dim0": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_4d_dim1": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_4d_dim2": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_4d_dim3": TestFailure(("lx_planning"), is_skip=True),
    "test_activation_fn_mish_fp16": TestFailure(("lx_planning"), is_skip=True),
    "test_activation_fn_silu_fp16": TestFailure(("lx_planning"), is_skip=True),
    "test_addmm_out_basic": TestFailure(("lx_planning"), is_skip=True),
}


def make_lx_planning_class(cls):
    return make_test_cls_with_patches(
        cls,
        "LxPlanning",
        "",
        (torch_spyre._inductor.config, "lx_planning", True),
        (torch_spyre._inductor.config, "allow_all_ops_in_lx_planning", True),
        (torch_spyre._inductor.config, "sencores", 1),
    )


class LxPlanningTwoOpPointwiseTest(unittest.TestCase):
    def compare_with_cpu(self, fn, *args, **kwargs):
        kwargs["cpu_compile"] = False

        @functools.wraps(fn)
        def make_seq_of_ops(*fn_args, **fn_kwargs):
            result = fn(*fn_args, **fn_kwargs)
            return pytree.tree_map(
                lambda x: x + x if isinstance(x, torch.Tensor) else x, result
            )

        return utils_inductor.compare_with_cpu(make_seq_of_ops, *args, **kwargs)

    def compare(
        self,
        fn,
        *args,
        atol=0.0,
        rtol=0.0,
        cpu_atol=0.1,
        cpu_rtol=0.1,
        needs_device=False,
    ):
        # utils_inductor.compare spyre with cpu and sendnn, here we skip sendnn
        @functools.wraps(fn)
        def make_seq_of_ops(*fn_args, **fn_kwargs):
            result = fn(*fn_args, **fn_kwargs)
            return pytree.tree_map(
                lambda x: x + x if isinstance(x, torch.Tensor) else x, result
            )

        return utils_inductor.compare_with_cpu(
            make_seq_of_ops,
            *args,
            atol=cpu_atol,
            rtol=cpu_rtol,
            needs_device=needs_device,
            cpu_compile=False,
        )


class LxPlanningTwoOpReductionTest(unittest.TestCase):
    def compare_with_cpu(self, fn, *args, **kwargs):
        kwargs["cpu_compile"] = False

        @functools.wraps(fn)
        def make_seq_of_ops(*fn_args, **fn_kwargs):
            result = fn(*fn_args, **fn_kwargs)
            return pytree.tree_map(
                lambda x: torch.sum(x, dim=0)
                if isinstance(x, torch.Tensor) and x.dtype == torch.float16
                else x,
                result,
            )

        return utils_inductor.compare_with_cpu(make_seq_of_ops, *args, **kwargs)

    def compare(
        self,
        fn,
        *args,
        atol=0.0,
        rtol=0.0,
        cpu_atol=0.1,
        cpu_rtol=0.1,
        needs_device=False,
    ):
        # utils_inductor.compare spyre with cpu and sendnn, here we skip sendnn
        @functools.wraps(fn)
        def make_seq_of_ops(*fn_args, **fn_kwargs):
            result = fn(*fn_args, **fn_kwargs)
            return pytree.tree_map(
                lambda x: torch.sum(x, dim=0)
                if isinstance(x, torch.Tensor) and x.dtype == torch.float16
                else x,
                result,
            )

        return utils_inductor.compare_with_cpu(
            make_seq_of_ops,
            *args,
            atol=cpu_atol,
            rtol=cpu_rtol,
            needs_device=needs_device,
            cpu_compile=False,
        )


copy_tests(
    make_lx_planning_class(inductor.test_inductor_ops.TestOps),
    LxPlanningTwoOpPointwiseTest,
    "lx_planning_pointwise",
    test_failures,
)
copy_tests(
    make_lx_planning_class(inductor.test_inductor_ops.TestOps),
    LxPlanningTwoOpReductionTest,
    "lx_planning_reduction",
    test_failures,
)

import functools
import copy
import torch_spyre
import os
import sys

from torch._dynamo.testing import make_test_cls_with_patches

import unittest

_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(_test_dir)

from inductor.test_inductor_ops import TestOps  # noqa: E402


def copy_tests(my_cls, other_cls, suffix, test_failures=None, xfail_prop=None):
    for name, value in my_cls.__dict__.items():
        if name.startswith("test_"):
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


def make_lx_planning_class(cls):
    return make_test_cls_with_patches(
        cls,
        "LxPlanning",
        "_lx_planning",
        (torch_spyre._inductor.config, "lx_planning", True),
    )


LxPlanningTemplate = make_lx_planning_class(TestOps)


class LxPlanningTest(unittest.TestCase):
    pass


copy_tests(LxPlanningTemplate, LxPlanningTest, "lx_planning")

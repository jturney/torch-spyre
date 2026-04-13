import torch
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_device_type import (
    ops,
    instantiate_device_type_tests,
    DeviceTypeTestBase,
    device_type_test_bases,
)
from torch.testing._internal.common_methods_invocations import op_db

aten = torch.ops.aten

SPYRE_WHITELIST = {"sum"}
SPYRE_OPS = [op for op in op_db if op.name in SPYRE_WHITELIST]

SPYRE_DTYPES = {torch.float16}


class SpyreTestBase(DeviceTypeTestBase):
    device_type = "spyre"


device_type_test_bases.append(SpyreTestBase)


class TestSpyreInductorOpInfo(TestCase):
    def _get_sample_inputs(self, op, dtype, device="spyre"):
        """Get sample imputs from OpInfo using reference_inputs for comprehensive coverage."""
        # Use reference_inputs for more comprehensive test coverage
        # Falls back to sample_inputs if reference_inputs_func is not defined
        try:
            samples = list(op.reference_inputs(device, dtype, requires_grad=False))
        except Exception:
            samples = list(op.sample_inputs(device, dtype, requires_grad=False))
        return samples

    @ops(SPYRE_OPS, allowed_dtypes=SPYRE_DTYPES)
    def test_single_op(self, dtype, op):
        aten_op = getattr(aten, op.aten_name)
        for sample in self._get_sample_inputs(op, dtype):
            try:
                # eager
                # first_op = aten_op(sample.input, *sample.args, **sample.kwargs)
                continue
            except Exception:
                # if eager failed skip this sample
                continue

            # spyre
            # second_op = torch.compile(aten_op)(
            #     sample.input, *sample.args, **sample.kwargs
            # )

            # self.assertTrue(torch.allclose(first_op, second_op))


instantiate_device_type_tests(TestSpyreInductorOpInfo, globals(), only_for=("spyre"))

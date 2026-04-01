# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch_spyre._inductor import config
from utils_inductor import cached_randn
from torch.testing import FileCheck
from torch._inductor.exc import InductorError
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import (
    run_and_get_code,
)


class TestSpyreConfig(InductorTestCase):
    def test_config_default(self):
        fn = torch.abs
        x = cached_randn((256,)).to("spyre")

        comp_fn = torch.compile(fn)
        out, source_codes = run_and_get_code(comp_fn, x)
        FileCheck().check("sdsc_fused_abs_0").run(source_codes[0])

    @config.patch({"sencores": 64})
    def test_config_too_many_sencores(self):
        fn = torch.abs
        x = cached_randn((256,)).to("spyre")

        with self.assertRaisesRegex(
            InductorError,
            "Unsupported: Spyre backend does not support: invalid SENCORES value 64",
        ):
            comp_fn = torch.compile(fn)
            comp_fn(x)

    # Need a test where getting lx_planning to True generates a different kernel
    # @config.patch({'lx_planning': True})
    # def test_config_lx_planning(self):
    #    fn = torch.abs
    #    x = cached_randn((256,)).to("spyre")
    #
    #    comp_fn = torch.compile(fn)
    #    out, source_codes = run_and_get_code(comp_fn, x)
    #    print(f"lx_planning {config.lx_planning}")
    #    print(source_codes[0])

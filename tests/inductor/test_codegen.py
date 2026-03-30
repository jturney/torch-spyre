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

from unittest.mock import patch as mock_patch
import torch
from torch_spyre._inductor import config
from torch_spyre._inductor.dsc import SuperDSCScheduling
from utils_inductor import cached_randn


def test_config_change_invalidates_cache():
    fn = torch.abs
    x = cached_randn((256,))
    call_count = 0
    original_define_kernel = SuperDSCScheduling.define_kernel

    def counting_define_kernel(self_sched, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_define_kernel(self_sched, *args, **kwargs)

    with mock_patch.object(SuperDSCScheduling, "define_kernel", counting_define_kernel):
        torch._dynamo.reset_code_caches()
        torch.compile(fn)(x.to("spyre"))
        first_count = call_count

        # Same config — should hit cache (no new codegen)
        # torch._dynamo.reset_code_caches()
        torch.compile(fn)(x.to("spyre"))
        assert call_count == first_count, "Expected cache hit"

        # Different config — should miss cache (new codegen)
        with config.patch("sencores", 1):
            torch._dynamo.reset_code_caches()
            torch.compile(fn)(x.to("spyre"))
            assert call_count > first_count, "Expected recompilation"

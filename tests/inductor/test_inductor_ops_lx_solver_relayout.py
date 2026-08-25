# Copyright 2026 The Torch-Spyre Authors.
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

"""LX planning sweep with the CP-SAT co-optimizing solver and in-solver relayout.

Re-wraps the two LX-planning wrap classes from
``test_inductor_ops_lx_planning.py`` with three additional config patches:

- ``layout_solver = "cpsat"``: joint core-division + LX placement solving
- ``co_optimizing_lx_planning = True``: divisions chosen with the cost model
- ``lx_solver_relayout = True``: relayout decisions made inside the solver

Everything else (the two-op wraps, canonical-subset selection via
``TEST_LX_PLANNING_FULL``, tolerances) is inherited from the LX-planning
wrapper, so any difference between this suite and the LX-planning suite is
attributable to the solver configuration alone.
"""

import os
import sys

import torch_spyre

from torch._dynamo.testing import make_test_cls_with_patches

_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(_test_dir)

import inductor.test_inductor_ops_lx_planning as _lx  # noqa: E402


def make_solver_relayout_class(cls):
    return make_test_cls_with_patches(
        cls,
        "SolverRelayout",
        "",
        (torch_spyre._inductor.config, "layout_solver", "cpsat"),
        (torch_spyre._inductor.config, "co_optimizing_lx_planning", True),
        (torch_spyre._inductor.config, "lx_solver_relayout", True),
    )


SolverRelayoutTwoOpPointwiseAdditionTest = make_solver_relayout_class(
    _lx.LxPlanningTwoOpPointwiseAdditionTest
)
SolverRelayoutTwoOpReductionTest = make_solver_relayout_class(
    _lx.LxPlanningTwoOpReductionTest
)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

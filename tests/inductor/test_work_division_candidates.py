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

"""Unit tests for ``enumerate_work_division_candidates``.

The enumerator must return every *permissible* core-division split for an op
(divisibility, ``<= max_cores``, ``<= MAX_SPAN_BYTES`` per-core span, ``<= 1``
reduction split). These tests run a real compile, intercept the post-layout
graph, and check the enumerated candidates against those invariants and against
the single split the legacy work-division passes actually committed.
"""

import math
import unittest
from contextlib import contextmanager
from typing import Callable, Optional
from unittest.mock import patch

import torch
from torch._inductor import config as t_inductor_config
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Pointwise, Reduction

from torch_spyre._inductor import config as ts_inductor_config
from torch_spyre._inductor import passes
from torch_spyre._inductor.passes import CustomPreSchedulingPasses
from torch_spyre._inductor.work_division import (
    MAX_SPAN_BYTES,
    collect_tensor_deps,
    enumerate_work_division_candidates,
    get_per_core_span,
)
from torch_spyre._inductor.pass_utils import (
    get_mem_deps_from_rw,
    iteration_space_from_op,
    splits_by_index_coeff,
)


class _PreSchedulingPassesWithVisitor(CustomPreSchedulingPasses):
    """Runs the normal pre-scheduling passes, then a test-supplied visitor.

    Mirrors the monkey-patch harness in ``test_scratchpad_use.py``: the visitor
    sees the graph after layouts *and* work division have been committed.
    """

    visitor: Optional[Callable[[GraphLowering], None]] = None

    def __call__(self, graph: GraphLowering) -> None:
        super().__call__(graph)
        if self.visitor is not None:
            self.visitor(graph)


class TestEnumerateWorkDivisionCandidates(unittest.TestCase):
    MAX_CORES = 8

    def setUp(self):
        torch.manual_seed(0xAFFE)
        self.patchers = [
            t_inductor_config.patch("force_disable_caches", True),
            ts_inductor_config.patch("sencores", self.MAX_CORES),
            patch.object(
                passes, "CustomPreSchedulingPasses", _PreSchedulingPassesWithVisitor
            ),
        ]
        for p in self.patchers:
            p.__enter__()
        torch.compiler.reset()

    def tearDown(self):
        _PreSchedulingPassesWithVisitor.visitor = None
        for p in reversed(self.patchers):
            p.__exit__(None, None, None)
        torch.compiler.reset()

    @contextmanager
    def _visit_ops(self, check: Callable[[ComputedBuffer], None]):
        def visitor(graph: GraphLowering) -> None:
            for op in graph.operations:
                if isinstance(op, ComputedBuffer) and isinstance(
                    op.data, (Pointwise, Reduction)
                ):
                    check(op)

        # staticmethod so assigning to the class attribute does not bind it as
        # a method (which would pass ``self`` as an extra positional arg).
        _PreSchedulingPassesWithVisitor.visitor = staticmethod(visitor)
        try:
            yield
        finally:
            _PreSchedulingPassesWithVisitor.visitor = None

    # -- helpers ----------------------------------------------------------

    def _assert_permissible(self, op: ComputedBuffer):
        """Every enumerated candidate must satisfy all four constraints."""
        candidates = enumerate_work_division_candidates(op, self.MAX_CORES)
        self.assertGreater(len(candidates), 0, f"{op.get_name()}: no candidates")

        it_space = iteration_space_from_op(op)
        args = get_mem_deps_from_rw(op.get_read_writes())
        input_tds, output_td = collect_tensor_deps(op, args)
        all_tds = input_tds + [output_td]
        coord_vars = {v for e in output_td.device_coords[:-1] for v in e.free_symbols}

        seen = set()
        for splits in candidates:
            # (2) core budget
            self.assertLessEqual(math.prod(splits.values()), self.MAX_CORES)
            # (4) at most one reduction (K) dim split
            n_red = sum(1 for v, s in splits.items() if s > 1 and v not in coord_vars)
            self.assertLessEqual(n_red, 1, f"{op.get_name()}: {n_red} K-splits")
            # (1) divisibility is guaranteed by enumerating divisors; (3) span
            for td in all_tds:
                self.assertLessEqual(
                    get_per_core_span(td, splits, it_space, {}), MAX_SPAN_BYTES
                )
            key = tuple(sorted((str(v), s) for v, s in splits.items()))
            self.assertNotIn(key, seen, f"{op.get_name()}: duplicate candidate")
            seen.add(key)

        # determinism: a second call returns the identical list
        self.assertEqual(
            candidates, enumerate_work_division_candidates(op, self.MAX_CORES)
        )

    def _assert_committed_is_member(self, op: ComputedBuffer):
        """The split the legacy passes committed must be an enumerated candidate."""
        committed = getattr(op, "op_it_space_splits", None)
        if committed is None:
            return  # op ran single-core; nothing committed
        candidates = enumerate_work_division_candidates(op, self.MAX_CORES)
        _, output_td = collect_tensor_deps(
            op, get_mem_deps_from_rw(op.get_read_writes())
        )
        rw = op.get_read_writes()
        write_index = output_td.dep.index
        first_read = next(iter(rw.reads), None)
        read_index = first_read.index if first_read is not None else write_index
        encoded = {
            repr(splits_by_index_coeff(c, write_index, read_index)) for c in candidates
        }
        self.assertIn(
            repr(committed),
            encoded,
            f"{op.get_name()}: committed split {committed} not in enumerated set",
        )

    def _run(self, fn, *args):
        def check(op):
            self._assert_permissible(op)
            self._assert_committed_is_member(op)

        with self._visit_ops(check):
            torch.compile(fn, fullgraph=True, dynamic=False)(*args)

    # -- tests ------------------------------------------------------------

    def test_pointwise(self):
        x = torch.randn(128, 256, dtype=torch.float16, device="spyre")
        y = torch.randn(128, 256, dtype=torch.float16, device="spyre")
        self._run(lambda a, b: a + b, x, y)

    def test_reduction(self):
        x = torch.randn(128, 256, dtype=torch.float16, device="spyre")
        self._run(lambda a: torch.softmax(a, dim=-1), x)

    def test_matmul(self):
        a = torch.randn(256, 128, dtype=torch.float16, device="spyre")
        b = torch.randn(128, 256, dtype=torch.float16, device="spyre")
        self._run(lambda x, y: x @ y, a, b)


if __name__ == "__main__":
    unittest.main()

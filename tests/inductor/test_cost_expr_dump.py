# Copyright 2025 IBM Corporation
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
"""The cost-expression dump (``SPYRE_DUMP_COST_EXPR_FILE``): the objective's
per-bundle terms, the solved symbol bindings and the evaluated prices, written
as one JSON record per co-optimized graph. No device, no graph: the record is
built from handcrafted buffers and terms."""

import json

import sympy

from torch_spyre._inductor.cost_model import CostParams
from torch_spyre._inductor.dump_common import emit_json_line
from torch_spyre._inductor.pass_utils import PerCoreView
from torch_spyre._inductor.scratchpad.allocator import CoOptimizingAllocator
from torch_spyre._inductor.scratchpad.lx_relayout import RelayoutCandidate
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision,
    CoreDivisionBuffer,
    RelayoutCharge,
    cost_expr_record,
    solved_bindings,
)

_CORE = sympy.Symbol("core_id")


def _view(slot: int) -> PerCoreView:
    return PerCoreView(((1, 4),), ((1, sympy.Mod(_CORE + slot, 4)),), 4)


def _buffers():
    p = CoreDivisionBuffer(
        "P",
        64,
        [0, 2],
        core_divisions=[CoreDivision(splits={1: 4}), CoreDivision(splits={0: 4})],
    )
    c = CoreDivisionBuffer(
        "C",
        64,
        [1, 2],
        core_divisions=[CoreDivision(splits={1: 4})],
        parents=["P"],
        cd_parent_relayouts={
            "P": [
                RelayoutCandidate(
                    parent="P",
                    consumer="C",
                    source_division=1,
                    consumer_division=0,
                    group=0,
                    source_view=_view(0),
                    destination_view=_view(1),
                    cost_ns=3000.0,
                    source_footprint_bytes=16,
                    destination_footprint_bytes=16,
                )
            ]
        },
    )
    return p, c


def test_solved_bindings_read_the_plan_like_the_annealer():
    p, c = _buffers()
    p.address, p.chosen_division = 0, 1
    c.address, c.chosen_division = None, 0
    b = solved_bindings([p, c])
    assert b[p.sym_is_lx] == 1 and b[c.sym_is_lx] == 0
    assert b[p.sym_division] == 1 and b[c.sym_division] == 0
    # division 1 of P splits axis 0 four ways and leaves axis 1 whole.
    splits = {str(k): b[sym] for k, sym in p.sym_core_divs.items()}
    assert set(splits.values()) == {4, 1}


def test_record_evaluates_every_term_under_the_solved_plan():
    p, c = _buffers()
    (copy,) = CoOptimizingAllocator._relayout_copy_buffers([p, c])
    p.address, p.chosen_division = 0, 1
    c.address, c.chosen_division = 16, 0
    copy.address = 32
    spill = 4000 * (1 - p.sym_is_lx) + 2000 * (1 - c.sym_is_lx)
    bundle_terms = [
        (["P"], 4000 * (1 - p.sym_is_lx)),
        (["C"], 2000 * (1 - c.sym_is_lx)),
    ]
    cost_expr = spill + copy.cost_term()
    rec = cost_expr_record(cost_expr, bundle_terms, [p, c, copy], CostParams())
    assert rec["buffers"] == ["P", "C"]
    assert [b["value_ns"] for b in rec["bundles"]] == [0.0, 0.0]
    (rt,) = rec["relayout_terms"]
    assert rt["source"] == "P" and rt["resident"] is True
    assert rt["value_ns"] == 3000.0, "P chose division 1, priced 3000 ns"
    assert rec["objective_ns"] == 3000.0
    assert rec["params"]["bw_peak_gbps"] == CostParams().bw_peak_gbps
    # srepr round-trips, including the RelayoutCharge node.
    back = sympy.parse_expr(rt["expr"], local_dict={"RelayoutCharge": RelayoutCharge})
    assert back.free_symbols == {copy.sym_is_lx, p.sym_division}


def test_emit_json_line_appends_one_record_per_call(tmp_path):
    path = tmp_path / "dump.jsonl"
    emit_json_line(str(path), {"a": 1})
    emit_json_line(str(path), {"b": sympy.Integer(2)})
    lines = path.read_text().splitlines()
    assert [json.loads(line) for line in lines] == [{"a": 1}, {"b": "2"}]

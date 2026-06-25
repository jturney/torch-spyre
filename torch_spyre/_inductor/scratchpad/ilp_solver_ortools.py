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

"""OR-Tools CP-SAT port of :class:`ILPLayoutSolver`.

A drop-in alternative to the Z3-based solver in ``ilp_solver.py``: same
constructor signature, same ``plan_layout(buffers) -> buffers`` contract,
operating on the same :class:`CoreDivisionBuffer`. The two are meant to be
swapped and compared -- the model encodes the *identical* problem (residency,
joint core-division, slicing-match gates, in-place merge relaxation, no-overlap
placement) and optimizes the *identical* lexicographic objective (minimize
spills, then maximize core occupancy).

Differences from the Z3 version are mechanical, not semantic:

* The two-phase ``check()`` search (seeded spill-budget scan, then binary
  occupancy search) collapses into a single weighted objective
  ``minimize(W * spills - occupancy)`` with ``W`` larger than any achievable
  occupancy, so spills strictly dominate -- CP-SAT proves the lexicographic
  optimum directly instead of via repeated solves.
* The reified disjunctions (no-overlap, slicing matches) use CP-SAT's
  ``OnlyEnforceIf`` / ``AddBoolOr`` instead of ``z3.Implies`` / ``z3.Or``.
* ``eff_size`` / ``cores`` are tied to the chosen division via ``AddElement``
  instead of a big disjunction.

The dependency handling is identical to the Z3 solver: in-place reuse is a
relaxation of pairwise no-overlap driven by each buffer's ``in_place_parents``
(``merge_vars``), and producer/consumer slicing consistency is enforced over
the ``parents`` edges using the precomputed ``cd_parent_matches`` pairs. The
deterministic post-step (``_justify``) and the placement-unit reconstruction
are shared with the Z3 solver, so a plan from either solver is compacted the
same way.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from ortools.sat.python import cp_model
else:
    try:
        from ortools.sat.python import cp_model

    except ImportError:  # pragma: no cover - exercised only when ortools is absent
        cp_model = None

from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivisionBuffer,
    MemoryPlanSolver,
    _assert_in_place_relationships,
)
from torch_spyre._inductor.scratchpad.ilp_solver import (
    ILPLayoutSolver,
    _PlacementUnit,
    _assert_core_divisions_enumerated,
)

__all__ = ["CpSatLayoutSolver"]

logger = logging.getLogger(__name__)


@dataclass
class _CoreDivisionBufferWithCpVars:
    """A :class:`CoreDivisionBuffer` bundled with the CP-SAT variables the solver
    creates for it, so one object flows through the solve instead of a buffer
    list shadowed by a parallel ``name -> {var}`` dict (the CP-SAT mirror of
    :class:`ilp_solver._CoreDivisionBufferWithVars`).

    The buffer spans ``[buffer.start_time, buffer.end_time)``; the vars encode
    where (``offset``) and whether (``in_buffer``) it resides in LX, and the
    chosen core division (``division``) with its per-core footprint
    (``eff_size``) and core occupancy (``cores``). ``merge_vars`` maps each
    in-place parent name to the merge bool for that parent->this edge.

    CP-SAT variables must be created against a model, so unlike the Z3 wrapper
    (whose vars are global symbols) this one takes the model and the unit
    capacity ``M`` and creates only the variables here; the constraints tying
    them together are added by the solver methods, exactly as in the Z3 path."""

    buffer: CoreDivisionBuffer
    model: "cp_model.CpModel"
    capacity_units: int

    def __post_init__(self):
        b = self.buffer
        m = self.model
        M = self.capacity_units
        self.name = b.name
        self.start_time = b.start_time
        self.end_time = b.end_time

        self.in_buffer = m.NewBoolVar(f"in_buffer_{b.name}")
        # offset domain [0, M-1]; the resident => offset+eff_size<=M bound is
        # added in the in-place relaxation pass (mirrors the Z3 box bound).
        self.offset = m.NewIntVar(0, max(0, M - 1), f"off_{b.name}")

        per_core = [
            int(np.ceil(b.size / cd.output_partition)) for cd in b.core_divisions
        ]
        partition = [cd.output_partition for cd in b.core_divisions]
        self.division = m.NewIntVar(0, len(b.core_divisions) - 1, f"div_{b.name}")
        self.eff_size = m.NewIntVar(0, max(per_core), f"eff_size_{b.name}")
        # cores this buffer occupies under chosen div
        self.cores = m.NewIntVar(0, max(partition), f"occ_{b.name}")
        self.merge_vars = {
            parent: m.NewBoolVar(f"merge_{parent}_{b.name}")
            for parent in b.in_place_parents
        }


class CpSatLayoutSolver(MemoryPlanSolver[CoreDivisionBuffer]):
    """LX placement via an OR-Tools CP-SAT search
    (``config.layout_solver == "cpsat"``).

    API-compatible with :class:`ILPLayoutSolver`; see the module docstring for
    the (mechanical-only) differences from the Z3 encoding. The problem and the
    lexicographic objective are identical, so on a given buffer set both solvers
    reach the same optimum spill count and occupancy.
    """

    def __init__(
        self,
        size: int,
        alignment: int = 128,
        time_limit_seconds: float = 10.0,
        bottom_justify: bool = True,
    ) -> None:
        if cp_model is None:
            raise ImportError(
                "The 'cpsat' layout solver requires the 'ortools' package, "
                "which is not installed. Install it with 'pip install ortools' "
                "or select a different layout_solver (e.g. 'greedy')."
            )
        super().__init__(size, alignment)
        # The solver works in alignment-sized units so every offset it picks is
        # automatically aligned; plan_layout scales sizes/offsets in and out.
        self._capacity_units = self.limit // self.alignment
        self._time_limit_seconds = time_limit_seconds
        self._bottom_justify = bottom_justify


    def plan_layout(
        self, buffers: list[CoreDivisionBuffer]
    ) -> list[CoreDivisionBuffer]:
        if not buffers:
            return []
        assert all(b.address is None for b in buffers), (
            "Buffers cannot be previously or partially planned"
        )
        _assert_in_place_relationships(buffers)
        _assert_core_divisions_enumerated(buffers)

        model = cp_model.CpModel()
        # Solve on copies so we never mutate the caller's buffers.
        working = {
            b.name: _CoreDivisionBufferWithCpVars(
                replace(b, size=int(np.ceil(b.size / self.alignment))),
                model,
                self._capacity_units,
            )
            for b in buffers
        }

        offsets, spilled, chosen_div = self._run(model, working)
        offsets = {k: v * self.alignment for k, v in offsets.items()}

        for b in buffers:
            b.address = None if b.name in spilled else offsets.get(b.name)
            b.chosen_division = chosen_div.get(b.name, b.chosen_division)
        return buffers

    # ------------------------------------------------------------------
    # Model build + solve
    # ------------------------------------------------------------------
    def _run(
        self,
        model: "cp_model.CpModel",
        tensors: dict[str, _CoreDivisionBufferWithCpVars],
    ) -> tuple[dict[str, int], set[str], dict[str, int]]:
        self._add_buffer_vars(model, tensors)
        self._add_inplace_relaxation(model, tensors)
        forced = self._add_core_division(model, tensors)
        self._add_objective(model, tensors)

        solver = cp_model.CpSolver()
        if self._time_limit_seconds:
            solver.parameters.max_time_in_seconds = float(self._time_limit_seconds)
        solver.parameters.num_search_workers = os.cpu_count() or 1
        # Fixed seed so a given worker configuration is reproducible run-to-run.
        solver.parameters.random_seed = 0
        status = solver.Solve(model)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[CP-SAT layout solver] tensors=%d forced spills=%d (%s) "
                "status=%s objective=%s walltime=%.2f ms",
                len(tensors),
                len(forced),
                ", ".join(sorted(forced)) or "none",
                solver.StatusName(status),
                solver.ObjectiveValue()
                if status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
                else "n/a",
                solver.WallTime() * 1e3,
            )

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError("CP-SAT memory planner found no feasible plan")

        return self._extract(solver, tensors)

    def _add_buffer_vars(
        self,
        model: "cp_model.CpModel",
        tensors: dict[str, _CoreDivisionBufferWithCpVars],
    ) -> None:
        """Tie each buffer's ``eff_size`` and ``cores`` to its chosen ``division``
        index. ``division`` indexes the candidate list, ``eff_size`` is the chosen
        division's per-core footprint (``size`` / its ``output_partition``), and
        ``cores`` its core occupancy. A non-re-divided buffer has a single
        candidate, so ``division`` is pinned to ``0`` and these are constants.
        Mirrors ``ILPLayoutSolver._add_buffer_vars``."""
        for sb in tensors.values():
            b = sb.buffer
            per_core = [
                int(np.ceil(b.size / cd.output_partition)) for cd in b.core_divisions
            ]
            partition = [cd.output_partition for cd in b.core_divisions]
            # tie effective size and core occupancy to the chosen division index
            model.AddElement(sb.division, per_core, sb.eff_size)
            model.AddElement(sb.division, partition, sb.cores)

    def _add_inplace_relaxation(
        self,
        model: "cp_model.CpModel",
        bufs: dict[str, _CoreDivisionBufferWithCpVars],
    ) -> None:
        """In-place reuse as a relaxation of the no-overlap constraint: each
        parent->child edge gets a merge bool that, when active, pins the pair to
        one shared base and lifts their pairwise no-overlap. Chains are induced
        transitively by the shared-offset equalities -- no merge groups, no path
        enumeration -- and ``_assert_in_place_relationships`` guarantees a
        parent/child overlap at exactly one tick, so only directly-edged pairs
        ever need the relaxation. The per-buffer ``merge_vars`` bools are read
        back in ``_extract`` to reconstruct placement units. Mirrors
        ``ILPLayoutSolver._add_inplace_relaxation``."""
        M = self._capacity_units

        # (src, dst) -> merge bools, consulted by the no-overlap relaxation.
        merge_between: dict[tuple[str, str], list] = {}
        # A storage slot is handed off linearly, so a buffer reuses at most one
        # parent and is reused by at most one child.
        incoming: dict[str, list] = {}
        outgoing: dict[str, list] = {}
        for dst, c in bufs.items():
            for src, edge in c.merge_vars.items():
                src_v, dst_v = bufs[src], bufs[dst]
                # active merge => shared base and both endpoints resident
                model.Add(src_v.offset == dst_v.offset).OnlyEnforceIf(edge)
                model.AddImplication(edge, src_v.in_buffer)
                model.AddImplication(edge, dst_v.in_buffer)
                # active merge => child reuses the parent's exact per-core storage,
                # so their chosen divisions must have equal per-core footprints.
                model.Add(dst_v.eff_size == src_v.eff_size).OnlyEnforceIf(edge)
                # active merge => parent and child must pick slicing-compatible divisions
                self._constrain_merge_division(model, bufs, src, dst, edge)
                merge_between.setdefault((src, dst), []).append(edge)
                outgoing.setdefault(src, []).append(edge)
                incoming.setdefault(dst, []).append(edge)

        for ms in (*incoming.values(), *outgoing.values()):
            if len(ms) > 1:
                model.AddAtMostOne(ms)

        for sb in bufs.values():
            # if a buffer is resident its top must be below the peak usage.
            model.Add(sb.offset + sb.eff_size <= M).OnlyEnforceIf(sb.in_buffer)

        self._apply_no_overlap_constraint(model, list(bufs.values()), merge_between)

    def _constrain_merge_division(
        self,
        model: "cp_model.CpModel",
        bufs: dict[str, _CoreDivisionBufferWithCpVars],
        src: str,
        dst: str,
        m,
    ) -> None:
        """Gate an in-place merge on slicing-compatible divisions: when ``m`` is
        active, parent (``src``) and child (``dst``) must pick divisions that
        induce the same per-core slicing of the shared storage. Uses the
        precomputed ``cd_parent_matches`` pairs (see ``_implicate_core_division``);
        no pairs => merge forbidden. Mirrors
        ``ILPLayoutSolver._constrain_merge_division``."""
        pv, cv = bufs[src], bufs[dst]
        compatible = bufs[dst].buffer.cd_parent_matches.get(src, [])
        self._gate_divisions(model, compatible, pv.division, cv.division, m)

    @staticmethod
    def _gate_divisions(model, compatible, src_div, dst_div, enforce_lit) -> None:
        """Enforce, when ``enforce_lit`` is true, that ``(src_div, dst_div)`` is
        one of the ``compatible`` (i, j) pairs. With no compatible pairs the
        relation is unsatisfiable, so ``enforce_lit`` is forced false -- the
        mirror of ``z3.Implies(m, z3.Or(pairs) if pairs else False)``."""
        if not compatible:
            model.Add(enforce_lit == 0)
            return
        pair_lits = []
        for i, j in compatible:
            lit = model.NewBoolVar("")
            model.Add(src_div == i).OnlyEnforceIf(lit)
            model.Add(dst_div == j).OnlyEnforceIf(lit)
            pair_lits.append(lit)
        model.AddBoolOr(pair_lits).OnlyEnforceIf(enforce_lit)

    @staticmethod
    def _apply_no_overlap_constraint(
        model: "cp_model.CpModel",
        bufs: list[_CoreDivisionBufferWithCpVars],
        merge_between: dict[tuple[str, str], list],
    ) -> None:
        """Pairwise no-overlap for time-overlapping buffers, relaxed by an active
        merge edge between the pair. Mirror of
        ``ILPLayoutSolver._apply_no_overlap_constraint``: for each resident pair
        either one clears the other in address space or a merge lets them share."""

        def time_overlap(
            a: _CoreDivisionBufferWithCpVars, b: _CoreDivisionBufferWithCpVars
        ) -> bool:
            return a.start_time < b.end_time and b.start_time < a.end_time

        for i in range(len(bufs)):
            for j in range(i + 1, len(bufs)):
                a, b = bufs[i], bufs[j]
                # unrelated if they never overlap in time
                if not time_overlap(a, b):
                    continue
                # An active merge edge (either direction) lifts the no-overlap so
                # the pair may share a base; else, if both resident, they must be
                # disjoint (a ends before b starts, or b before a).
                relax = merge_between.get((a.name, b.name), []) + merge_between.get(
                    (b.name, a.name), []
                )
                a_left = model.NewBoolVar("")
                b_left = model.NewBoolVar("")
                model.Add(a.offset + a.eff_size <= b.offset).OnlyEnforceIf(a_left)
                model.Add(b.offset + b.eff_size <= a.offset).OnlyEnforceIf(b_left)
                # not-resident(a) | not-resident(b) | a_left | b_left | merge...
                model.AddBoolOr(
                    [a.in_buffer.Not(), b.in_buffer.Not(), a_left, b_left] + list(relax)
                )

    def _get_children(
        self, bufs: dict[str, _CoreDivisionBufferWithCpVars]
    ) -> dict[str, list[tuple[str, list[tuple[int, int]]]]]:
        """parent name -> list of (child name, match_pairs), where ``match_pairs``
        is the child's ``cd_parent_matches[parent]`` (empty when the edge has no
        compatible division). The child's ``parents`` define the edges. Mirrors
        ``ILPLayoutSolver._get_children``."""
        children_of: dict[str, list[tuple[str, list[tuple[int, int]]]]] = {}
        for sb in bufs.values():
            t = sb.buffer
            for parent in t.parents:
                children_of.setdefault(parent, []).append(
                    (t.name, t.cd_parent_matches.get(parent, []))
                )
        return children_of

    def _trim_oversized_tensors(
        self,
        model: "cp_model.CpModel",
        bufs: dict[str, _CoreDivisionBufferWithCpVars],
    ) -> set[str]:
        """Pin out of LX the buffers whose non-residency is fixed up front: those
        whose *smallest* candidate footprint still exceeds capacity, and those
        marked ``residency_allowed=False``. Mirrors
        ``ILPLayoutSolver._trim_oversized_tensors``."""
        forced = set()
        for sb in bufs.values():
            t = sb.buffer
            min_size = min(
                int(np.ceil(t.size / cd.output_partition)) for cd in t.core_divisions
            )
            if min_size > self._capacity_units or not t.residency_allowed:
                forced.add(t.name)
                model.Add(sb.in_buffer == 0)
        return forced

    def _implicate_core_division(
        self,
        model: "cp_model.CpModel",
        children_of: dict[str, list[tuple[str, list[tuple[int, int]]]]],
        bufs: dict[str, _CoreDivisionBufferWithCpVars],
    ) -> None:
        """Slicing-consistency gate: a resident buffer's division must match
        *every* consumer's division under the ``cd_parent_matches`` pairs. A
        buffer with no consumer edge, or with a consumer that has no compatible
        pair, can never reside. Mirrors
        ``ILPLayoutSolver._implicate_core_division``."""
        for sb in bufs.values():
            t = sb.buffer
            kids = children_of.get(t.name, [])
            if not kids:
                # Nothing consumes this buffer from LX -> it can never reside.
                model.Add(sb.in_buffer == 0)
                continue
            for _child, compatible in kids:
                if not compatible:
                    # This child can never match -> the buffer cannot reside.
                    model.Add(sb.in_buffer == 0)
                    continue
                pair_lits = []
                for i, j in compatible:
                    lit = model.NewBoolVar("")
                    model.Add(sb.division == i).OnlyEnforceIf(lit)
                    model.Add(bufs[_child].division == j).OnlyEnforceIf(lit)
                    pair_lits.append(lit)
                model.AddBoolOr(pair_lits).OnlyEnforceIf(sb.in_buffer)

    def _add_core_division(
        self,
        model: "cp_model.CpModel",
        bufs: dict[str, _CoreDivisionBufferWithCpVars],
    ) -> set[str]:
        """Wire up children_of, forced spills, and the slicing-match gate.
        Returns the forced-spill set (for debug logging). Matching is driven
        entirely by the precomputed ``cd_parent_matches`` pairs. Mirrors
        ``ILPLayoutSolver._add_core_division``."""
        children_of = self._get_children(bufs)
        forced = self._trim_oversized_tensors(model, bufs)
        self._implicate_core_division(model, children_of, bufs)
        return forced

    def _add_objective(
        self,
        model: "cp_model.CpModel",
        bufs: dict[str, _CoreDivisionBufferWithCpVars],
    ) -> None:
        """Single weighted lexicographic objective: minimize spills, then
        maximize total resident core occupancy. ``W`` exceeds any achievable
        occupancy, so a single spill always outweighs any occupancy gain --
        identical optimum to the Z3 solver's two-phase search."""
        max_occ = sum(
            max(cd.output_partition for cd in sb.buffer.core_divisions)
            for sb in bufs.values()
        )
        W = max_occ + 1

        spill = sum(1 - sb.in_buffer for sb in bufs.values())

        occ_terms = []
        for sb in bufs.values():
            occ_res = model.NewIntVar(
                0, max(cd.output_partition for cd in sb.buffer.core_divisions), ""
            )
            model.Add(occ_res == sb.cores).OnlyEnforceIf(sb.in_buffer)
            model.Add(occ_res == 0).OnlyEnforceIf(sb.in_buffer.Not())
            occ_terms.append(occ_res)

        model.Minimize(W * spill - sum(occ_terms))

    # ------------------------------------------------------------------
    # Extract (shares _PlacementUnit / _justify with the Z3 solver)
    # ------------------------------------------------------------------
    def _extract(
        self,
        solver: "cp_model.CpSolver",
        bufs: dict[str, _CoreDivisionBufferWithCpVars],
    ) -> tuple[dict[str, int], set[str], dict[str, int]]:
        """Read the solution into (offsets, spilled, chosen_div). When
        bottom_justify is set, slide each placement unit down to the lowest free
        address (preserving in-place merges, never raising the peak). Mirrors
        ``ILPLayoutSolver._extract``."""
        by_name = {name: sb.buffer for name, sb in bufs.items()}
        spilled = {
            name for name, sb in bufs.items() if not solver.BooleanValue(sb.in_buffer)
        }
        chosen_div = {name: solver.Value(sb.division) for name, sb in bufs.items()}

        def footprint(t: CoreDivisionBuffer) -> int:
            return int(
                np.ceil(t.size / t.core_divisions[chosen_div[t.name]].output_partition)
            )

        if not self._bottom_justify:
            return (
                {
                    name: solver.Value(sb.offset)
                    for name, sb in bufs.items()
                    if solver.BooleanValue(sb.in_buffer)
                },
                spilled,
                chosen_div,
            )

        # A placement unit is a connected component of active merge edges: its
        # members share one base (the merge equalities), so the component slides
        # as a single block and in-place reuse is preserved.
        resident = [n for n in by_name if n not in spilled]
        parent = {n: n for n in resident}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for dst, c in bufs.items():
            for src, edge in c.merge_vars.items():
                if solver.BooleanValue(edge):
                    parent[find(src)] = find(dst)

        components: dict[str, list[str]] = {}
        for n in resident:
            components.setdefault(find(n), []).append(n)

        units = [
            _PlacementUnit(
                members=names,
                footprint=max(footprint(by_name[n]) for n in names),
                start_time=min(by_name[n].start_time for n in names),
                end_time=max(by_name[n].end_time for n in names),
                original_offset=solver.Value(bufs[names[0]].offset),
            )
            for names in components.values()
        ]
        return ILPLayoutSolver._justify(units), spilled, chosen_div

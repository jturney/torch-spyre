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

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any, Optional, TYPE_CHECKING
import numpy as np
import math


if TYPE_CHECKING:
    import z3
else:
    try:
        import z3

    except ImportError:  # pragma: no cover - exercised only when z3 is absent
        z3 = None

from torch_spyre._inductor.scratchpad.plan_solver import (
    LifetimeBoundBuffer,
    MemoryPlanSolver,
    _assert_in_place_relationships,
)

__all__ = ["CoreDivisionBuffer", "ILPLayoutSolver"]

logger = logging.getLogger(__name__)


@dataclass
class CoreDivision:
    """One permissible core-division of a buffer's producing op.

    ``output_splits`` / ``reduction_splits`` are the stride/coeff-keyed encoding
    produced by :func:`pass_utils.splits_by_index_coeff` -- exactly the shape
    stored in ``op.op_it_space_splits``. ``ILPLayoutSolver`` uses these to size
    the buffer (per-core footprint = total / ``output_partition``).
    """

    output_splits: dict[int, int] = field(default_factory=dict)
    reduction_splits: dict[int, int] = field(default_factory=dict)

    @property
    def cores_used(self) -> int:
        return math.prod(self.output_splits.values()) * math.prod(
            self.reduction_splits.values()
        )

    @property
    def is_clean(self) -> bool:
        """True when no reduction axis is split, so the output is fully sliced
        across cores (no per-core partial sums)."""
        return not self.reduction_splits

    @property
    def output_partition(self) -> int:
        """How many cores the output buffer is sliced across."""
        return math.prod(self.output_splits.values())

    def signature_key(self):
        """Per-core slicing signature, or ``None`` for a reduction-split division
        (a ``None`` never compares equal, so partial-reduction divisions never
        match)."""
        return tuple(sorted(self.output_splits.items())) if self.is_clean else None

    @property
    def label(self) -> str:
        out = ",".join(f"s{s}/{f}" for s, f in sorted(self.output_splits.items()))
        red = ",".join(f"~s{s}/{f}" for s, f in sorted(self.reduction_splits.items()))
        return " ".join(p for p in (out, red) if p) or "whole"


@dataclass
class CoreDivisionBuffer(LifetimeBoundBuffer):
    """A :class:`LifetimeBoundBuffer` carrying the joint core-division metadata
    consumed only by :class:`ILPLayoutSolver`.

    The placement-only solvers (greedy/first-fit/best-fit) never look at these
    fields, so they stay on this subclass rather than the shared base.
    """

    core_divisions: list[CoreDivision] = field(default_factory=list)
    # Producer buffer names; defines the producer->consumer edges for matching.
    cd_parents: list[str] = field(default_factory=list[str])
    # parent_buf_name -> (parent_div_idx, this_div_idx) pairs that induce the
    # *same per-core slicing of the parent*, precomputed by the allocator via
    # ``_per_core_view_on_buf`` (physical device-dim view equality, correct
    # across reductions/reshapes). These are the sole slicing-match predicate;
    # an absent/empty entry means no compatible division, so the gate forbids
    # the merge/residency across that edge.
    cd_parent_matches: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    chosen_division: Optional[int] = None
    # When False the buffer may not be made resident (e.g. a graph boundary that
    # can't be LX-pinned without a clone). It is still handed to the solver so it
    # participates in matching -- a forced-out consumer keeps its producers'
    # residency viable instead of orphaning them.
    residency_allowed: bool = True


@dataclass
class _Box:
    """A buffer's memory layout spanning [start_time, end_time)."""

    name: str
    start_time: int
    end_time: int
    goff: z3.ArithRef
    gsize: z3.ArithRef
    var: z3.BoolRef


@dataclass
class _InPlaceCandidate:
    src: str
    dst: str


@dataclass
class _PlacementUnit:
    """A connected component of in-place-merged buffers placed as one block."""

    members: list[str]
    footprint: int
    start_time: int
    end_time: int
    base0: int  # offset z3 chose, before bottom-justify
    base: int = 0  # final justified offset


def _assert_core_divisions_enumerated(buffers: list[CoreDivisionBuffer]):
    """Assert that all buffers have enumerated core divisions."""
    for b in buffers:
        assert len(b.core_divisions) != 0, (
            "All buffers must have at least 1 valid core division"
        )


class ILPLayoutSolver(MemoryPlanSolver[CoreDivisionBuffer]):
    """LX placement via a Z3 satisfiability search
    (``config.layout_solver == "ilp"``).

    Joint core-division: ``size`` is the *total* device footprint; the solver
    picks each buffer's division via a ``div`` var indexing the candidate list
    (from ``enumerate_work_division_candidates``), sizes it by the chosen
    division's ``output_partition``, and constrains a resident buffer's
    producer/consumer divisions to the same per-core slicing
    (``_implicate_core_division``).

    The search is satisfiability, not optimization. The forced set (buffers too
    large to ever fit) is an a-priori lower bound on spills, so we seed the spill
    budget there and relax upward only when infeasible; the first feasible
    ``check()`` is usually already optimal. ``bottom_justify=True`` then slides
    each placement unit down to the lowest free address, closing float gaps
    without raising the peak.

    In-place reuse is a *relaxation* of the no-overlap constraint: each
    parent/child pair gets a merge bool that, when active, pins the pair to one
    base and exempts it from pairwise no-overlap. Chains fall out transitively
    from the shared-offset equalities -- no path enumeration. The single-tick
    lifetime-overlap invariant (``_assert_in_place_relationships``) means only
    directly-edged buffers ever overlap in time, so the per-edge relaxation is
    exact.
    """

    def __init__(
        self,
        size: int,
        alignment: int = 128,
        time_limit_seconds: float = 10.0,
        bottom_justify: bool = True,
    ) -> None:
        if z3 is None:
            raise ImportError(
                "The 'ilp' layout solver requires the 'z3-solver' package, "
                "which is not installed. Install it with 'pip install z3-solver' "
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

        candidates = [
            _InPlaceCandidate(src=parent, dst=b.name)
            for b in buffers
            for parent in b.in_place_parents
        ]

        # Solve on copies so we never mutate the caller's buffers.
        working = [
            replace(
                b,
                size=int(np.ceil(b.size / self.alignment)),
            )
            for b in buffers
        ]

        offsets, spilled, chosen_div = self._run(working, candidates)
        offsets = {k: v * self.alignment for k, v in offsets.items()}

        for b in buffers:
            b.address = None if b.name in spilled else offsets.get(b.name)
            b.chosen_division = chosen_div.get(b.name, b.chosen_division)
        return buffers

    def _run(
        self,
        tensors: list[CoreDivisionBuffer],
        candidates: list[_InPlaceCandidate],
    ) -> tuple[dict[str, int], set[str], dict[str, int]]:
        by_name = {t.name: t for t in tensors}
        self._check_candidates(by_name, candidates)

        opt = z3.Solver()
        buffer_vars = self._add_buffer_vars(opt, tensors)
        edges = self._add_inplace_relaxation(opt, tensors, candidates, buffer_vars)
        forced = self._add_core_division(opt, tensors, buffer_vars)
        model = self._search(opt, tensors, buffer_vars, forced)
        return self._extract(model, tensors, buffer_vars, edges)

    @staticmethod
    def _check_candidates(
        by_name: dict[str, CoreDivisionBuffer],
        candidates: list[_InPlaceCandidate],
    ) -> None:
        for c in candidates:
            if c.src not in by_name or c.dst not in by_name:
                raise ValueError(f"Candidate {c} references unknown tensor")

    @staticmethod
    def _apply_no_overlap_constraint(
        opt: z3.Solver,
        boxes: list[_Box],
        merge_between: dict[tuple[str, str], list[z3.BoolRef]],
    ) -> None:
        def time_overlap(a: _Box, b: _Box) -> bool:
            return a.start_time < b.end_time and b.start_time < a.end_time

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                # unrelated if they never overlap in time
                if not time_overlap(a, b):
                    continue
                # An active merge edge (either direction) lifts the no-overlap so
                # the pair may share a base; else, if both resident, they must be
                # disjoint (a ends before b starts, or b before a).
                relax = merge_between.get((a.name, b.name), []) + merge_between.get(
                    (b.name, a.name), []
                )
                opt.add(
                    z3.Implies(
                        z3.And(a.var, b.var),
                        z3.Or(
                            a.goff + a.gsize <= b.goff,
                            b.goff + b.gsize <= a.goff,
                            *relax,
                        ),
                    )
                )

    def _add_buffer_vars(
        self, opt: z3.Solver, tensors: list[CoreDivisionBuffer]
    ) -> dict[str, dict[str, z3.ExprRef]]:
        """Allocate per-tensor vars: name -> {offset, in_buffer, div_var,
        eff_size, occ}. ``div_var`` indexes the candidate list, ``eff_size`` is
        the chosen division's per-core footprint (``size`` / its
        ``output_partition``), and ``occ`` its core occupancy. A non-re-divided
        buffer has a single candidate, so ``div_var`` is ``[0, 0]`` and these are
        constants."""
        buffer_vars: dict[str, dict[str, z3.ExprRef]] = {}
        for t in tensors:
            n = t.name
            entry: dict[str, z3.ExprRef] = {}
            buffer_vars[n] = entry

            entry["in_buffer"] = z3.Bool(f"in_buf_{n}")  # is buffer in lx?
            offset = z3.Int(f"off_{n}")
            opt.add(offset >= 0, offset < self._capacity_units)
            entry["offset"] = offset  # where is the buffer in lx?

            dv = z3.Int(f"div_{n}")
            opt.add(dv >= 0, dv <= len(t.core_divisions) - 1)
            entry["div_var"] = dv  # which core-division index are we using?

            per_core = [
                int(np.ceil(t.size / cd.output_partition)) for cd in t.core_divisions
            ]
            partition = [cd.output_partition for cd in t.core_divisions]
            sv = z3.Int(f"size_{n}")
            ov = z3.Int(f"occ_{n}")  # cores this buffer occupies under chosen div
            opt.add(
                z3.Or(
                    [
                        z3.And(dv == i, sv == per_core[i], ov == partition[i])
                        for i in range(len(per_core))
                    ]
                )
            )  # tie effective size and core occupancy to the chosen division index
            entry["eff_size"] = sv
            entry["occ"] = ov
        return buffer_vars

    def _add_inplace_relaxation(
        self,
        opt: z3.Solver,
        tensors: list[CoreDivisionBuffer],
        candidates: list[_InPlaceCandidate],
        buffer_vars: dict[str, dict[str, z3.ExprRef]],
    ) -> list[tuple[str, str, z3.BoolRef]]:
        """In-place reuse as a relaxation of the no-overlap constraint: each
        parent->child edge gets a merge bool that, when active, pins the pair to
        one shared base and lifts their pairwise no-overlap. Chains are induced
        transitively by the shared-offset equalities -- no merge groups, no path
        enumeration -- and ``_assert_in_place_relationships`` guarantees a
        parent/child overlap at exactly one tick, so only directly-edged pairs
        ever need the relaxation. Returns the ``(src, dst, merge_bool)`` edge
        list for placement-unit reconstruction in ``_extract``."""
        M = self._capacity_units

        edges: list[tuple[str, str, z3.BoolRef]] = []
        # (src, dst) -> merge bools, consulted by the no-overlap relaxation.
        merge_between: dict[tuple[str, str], list[z3.BoolRef]] = {}
        # A storage slot is handed off linearly, so a buffer reuses at most one
        # parent and is reused by at most one child.
        incoming: dict[str, list[z3.BoolRef]] = {}
        outgoing: dict[str, list[z3.BoolRef]] = {}
        for c in candidates:
            m = z3.Bool(f"merge_{c.src}_{c.dst}")
            edges.append((c.src, c.dst, m))
            src_v, dst_v = buffer_vars[c.src], buffer_vars[c.dst]
            # active merge => shared base and both endpoints resident
            opt.add(z3.Implies(m, src_v["offset"] == dst_v["offset"]))
            opt.add(z3.Implies(m, src_v["in_buffer"]))
            opt.add(z3.Implies(m, dst_v["in_buffer"]))
            # active merge => child reuses the parent's exact per-core storage,
            # so their chosen divisions must have equal per-core footprints.
            opt.add(z3.Implies(m, dst_v["eff_size"] == src_v["eff_size"]))
            # active merge => parent and child must pick slicing-compatible divisions
            self._constrain_merge_division(opt, tensors, buffer_vars, c, m)
            merge_between.setdefault((c.src, c.dst), []).append(m)
            outgoing.setdefault(c.src, []).append(m)
            incoming.setdefault(c.dst, []).append(m)

        for ms in (*incoming.values(), *outgoing.values()):
            if len(ms) > 1:
                opt.add(z3.Sum(ms) <= 1)

        # One box per buffer: residency drives peak, and the pairwise no-overlap
        # may be relaxed by an active merge edge between the pair.
        boxes = [
            _Box(
                name=t.name,
                start_time=t.start_time,
                end_time=t.end_time,
                goff=buffer_vars[t.name]["offset"],
                gsize=buffer_vars[t.name]["eff_size"],
                var=buffer_vars[t.name]["in_buffer"],
            )
            for t in tensors
        ]
        for b in boxes:
            # if a buffer is resident its top must be below the peak usage.
            opt.add(z3.Implies(b.var, b.goff + b.gsize <= M))

        self._apply_no_overlap_constraint(opt, boxes, merge_between)
        return edges

    def _constrain_merge_division(
        self,
        opt: z3.Solver,
        tensors: list[CoreDivisionBuffer],
        buffer_vars: dict[str, dict[str, z3.ExprRef]],
        c: _InPlaceCandidate,
        m: z3.BoolRef,
    ) -> None:
        """Gate an in-place merge on slicing-compatible divisions: when ``m`` is
        active, parent (``c.src``) and child (``c.dst``) must pick divisions that
        induce the same per-core slicing of the shared storage. Uses the
        precomputed ``cd_parent_matches`` pairs (see ``_implicate_core_division``);
        no pairs => merge forbidden. A fixed-division endpoint has no ``div_var``
        to constrain and is already governed by the ``eff_size`` equality.
        """
        pv, cv = buffer_vars[c.src], buffer_vars[c.dst]
        by_name = {t.name: t for t in tensors}
        child = by_name[c.dst]
        compatible = child.cd_parent_matches.get(c.src, [])
        pairs = [
            z3.And(pv["div_var"] == i, cv["div_var"] == j) for (i, j) in compatible
        ]
        opt.add(z3.Implies(m, z3.Or(pairs) if pairs else z3.BoolVal(False)))

    def _get_children(
        self, tensors: list[CoreDivisionBuffer]
    ) -> dict[str, list[tuple[str, list[tuple[int, int]]]]]:
        """parent name -> list of (child name, match_pairs), where ``match_pairs``
        is the child's ``cd_parent_matches[parent]`` (empty when the edge has no
        compatible division). The child's ``cd_parents`` define the edges."""
        children_of: dict[str, list[tuple[str, Any]]] = {}
        for t in tensors:
            for parent in t.cd_parents:
                children_of.setdefault(parent, []).append(
                    (
                        t.name,
                        t.cd_parent_matches.get(parent, []),
                    )
                )
        return children_of

    def _trim_oversized_tensors(
        self,
        opt: z3.Solver,
        tensors: list[CoreDivisionBuffer],
        buffer_vars: dict[str, dict[str, z3.ExprRef]],
    ) -> set[str]:
        """Pin out of LX the buffers whose non-residency is fixed up front:
        those whose *smallest* candidate footprint still exceeds capacity, and
        those marked ``residency_allowed=False`` (e.g. graph boundaries). This
        set is the search's lower-bound spill seed. Division-dependent residency
        (no consumer, or all consumers mismatch the slicing) is left to
        ``_implicate_core_division`` rather than decided here."""
        forced = set()
        for t in tensors:
            min_size = (
                min(
                    int(np.ceil(t.size / cd.output_partition))
                    for cd in t.core_divisions
                )
                if t.core_divisions
                else t.size
            )
            if min_size > self._capacity_units or not t.residency_allowed:
                forced.add(t.name)
                opt.add(z3.Not(buffer_vars[t.name]["in_buffer"]))
        return forced

    def _implicate_core_division(
        self,
        opt: z3.Solver,
        tensors: list[CoreDivisionBuffer],
        children_of: dict[str, list[tuple[str, list[tuple[int, int]]]]],
        buffer_vars: dict[str, dict[str, z3.ExprRef]],
    ) -> None:
        """Slicing-consistency gate: a resident buffer's division must match
        *every* consumer's division under the ``cd_parent_matches`` pairs. The
        buffer carries one ``div_var``, so requiring all consumers to agree pins
        them to a single per-core slicing; a buffer with diverging consumer
        demands is forced out of LX and re-sliced per consumer on load from main
        memory. An edge with no compatible pair never matches, so a
        partial-reduction buffer resides only when a consumer reads those
        partials under the identical slicing. A divided buffer with no consumer
        edge is gated out the same way -- nothing reads it from LX."""
        for t in tensors:
            kids = children_of.get(t.name, [])
            if not kids:
                # Nothing consumes this buffer from LX -> it can never reside.
                opt.add(z3.Not(buffer_vars[t.name]["in_buffer"]))
                continue
            child_matches = []
            for _child, compatible in kids:
                # ``cd_parent_matches`` pairs are the match predicate (physical
                # device-dim view equality, correct across reductions/reshapes
                # where a coeff signature would conflate slicings). Empty => no
                # compatible division => no match, forbidding residency.
                pairs = [
                    z3.And(
                        buffer_vars[t.name]["div_var"] == i,
                        buffer_vars[_child]["div_var"] == j,
                    )
                    for (i, j) in compatible
                ]
                child_matches.append(z3.Or(pairs) if pairs else z3.BoolVal(False))
            opt.add(z3.Implies(buffer_vars[t.name]["in_buffer"], z3.And(child_matches)))

    def _add_core_division(
        self,
        opt: z3.Solver,
        tensors: list[CoreDivisionBuffer],
        buffer_vars: dict[str, dict[str, z3.ExprRef]],
    ) -> set[str]:
        """Wire up children_of, forced spills, and the slicing-match gate.
        Returns the forced-spill set (the search's lower-bound seed). Matching is
        driven entirely by the precomputed ``cd_parent_matches`` pairs."""
        children_of = self._get_children(tensors)
        forced = self._trim_oversized_tensors(opt, tensors, buffer_vars)
        self._implicate_core_division(opt, tensors, children_of, buffer_vars)
        return forced

    def _search(
        self,
        opt: z3.Solver,
        tensors: list[CoreDivisionBuffer],
        buffer_vars: dict[str, dict[str, z3.ExprRef]],
        forced: set[str],
    ) -> z3.ModelRef:
        """Two sequential satisfiability phases -- residency, then occupancy --
        so the second never trades away the first, avoiding a full
        ``z3.Optimize`` proof.

        Phase 1 (residency): spill nothing beyond ``forced``, relaxing the budget
        upward only if infeasible. Seeded at ``len(forced)``, so the first
        ``check()`` is usually optimal; the winning budget is then pinned.

        Phase 2 (occupancy): with spills fixed, binary-search the largest
        feasible total core occupancy (sum of resident ``output_partition``s) --
        the most resident-but-finely-split plan wins. With no real divisions
        every ``output_partition`` is 1, so occupancy is already pinned and the
        phase converges immediately."""
        if self._time_limit_seconds:
            opt.set("timeout", int(self._time_limit_seconds * 1000))

        spill_count = z3.Sum(
            [z3.If(buffer_vars[t.name]["in_buffer"], 0, 1) for t in tensors]
        )

        n_tensors = len(tensors)
        lo = len(forced)
        iterations = []  # (budget, status, seconds)
        model = None
        won_budget = None
        t_start = time.perf_counter()
        for budget in range(lo, n_tensors + 1):
            opt.push()
            opt.add(spill_count <= budget)
            t0 = time.perf_counter()
            status = opt.check()
            iterations.append((budget, status, time.perf_counter() - t0))
            if status == z3.sat:
                model = opt.model()
                won_budget = budget
                opt.pop()
                break
            opt.pop()

        # Phase 2: pin the optimal spill budget, then push occupancy as high as
        # it will go without spilling anything more.
        won_occupancy = None
        occ_iters = 0
        occ_iterations = []  # (mid, status, seconds)
        if model is not None:
            opt.add(spill_count == won_budget)
            occ_terms = [
                z3.If(v["in_buffer"], v["occ"], 0)
                for v in buffer_vars.values()
                if "occ" in v
            ]
            if occ_terms:
                occupancy = z3.Sum(occ_terms)
                lo_occ = model.eval(occupancy, model_completion=True).as_long()
                hi_occ = sum(
                    max(cd.output_partition for cd in t.core_divisions)
                    for t in tensors
                    if t.core_divisions
                )
                won_occupancy = lo_occ
                while lo_occ < hi_occ:
                    mid = (lo_occ + hi_occ + 1) // 2
                    opt.push()
                    opt.add(occupancy >= mid)
                    occ_iters += 1
                    t0 = time.perf_counter()
                    status = opt.check()
                    occ_iterations.append((mid, status, time.perf_counter() - t0))
                    if status == z3.sat:
                        model = opt.model()
                        won_occupancy = mid
                        lo_occ = mid
                    else:
                        hi_occ = mid - 1
                    opt.pop()
        total = time.perf_counter() - t_start

        ######################################
        # Debug: search timing / iterations
        ######################################
        if logger.isEnabledFor(logging.DEBUG):
            lines = [
                "[ILP layout solver]",
                f"  tensors            : {n_tensors}",
                f"  forced spills      : {lo} ({', '.join(sorted(forced)) or 'none'})",
                f"  budget seed / max  : {lo} / {n_tensors}",
                f"  time limit / check : {self._time_limit_seconds}s",
                f"  residency iters    : {len(iterations)}",
            ]
            for i, (budget, status, secs) in enumerate(iterations):
                lines.append(
                    f"    [{i}] spill<={budget:<4d} {str(status):>7s} {secs * 1e3:9.2f} ms"
                )
            outcome = (
                f"SAT @ spill<={won_budget}, occupancy={won_occupancy}"
                if model is not None
                else "NO FEASIBLE PLAN"
            )
            lines.append(f"  occupancy iters    : {occ_iters} (won={won_occupancy})")
            for i, (mid, status, secs) in enumerate(occ_iterations):
                lines.append(
                    f"    [{i}] occ>={mid:<6d} {str(status):>7s} {secs * 1e3:9.2f} ms"
                )
            lines.append(f"  result             : {outcome}")
            lines.append(f"  total search time  : {total * 1e3:.2f} ms")
            logger.debug("\n".join(lines))

        if model is None:
            raise RuntimeError("ILP memory planner found no feasible plan")
        return model

    def _extract(
        self,
        model: z3.ModelRef,
        tensors: list[CoreDivisionBuffer],
        buffer_vars: dict[str, dict[str, z3.ExprRef]],
        edges: list[tuple[str, str, z3.BoolRef]],
    ) -> tuple[dict[str, int], set[str], dict[str, int]]:
        """Read the model into (offsets, spilled, chosen_div). When
        bottom_justify is set, slide each placement unit down to the lowest free
        address (preserving in-place merges, never raising the peak)."""

        def bval(b):
            return z3.is_true(model.eval(b, model_completion=True))

        def ival(x):
            return model.eval(x, model_completion=True).as_long()

        by_name = {t.name: t for t in tensors}
        spilled = {
            t.name for t in tensors if not bval(buffer_vars[t.name]["in_buffer"])
        }
        chosen_div = {
            n: ival(v["div_var"]) for n, v in buffer_vars.items() if "div_var" in v
        }

        def footprint(t: CoreDivisionBuffer) -> int:
            if t.core_divisions:
                return int(
                    np.ceil(
                        t.size / t.core_divisions[chosen_div[t.name]].output_partition
                    )
                )
            return t.size

        if not self._bottom_justify:
            return (
                {
                    t.name: ival(buffer_vars[t.name]["offset"])
                    for t in tensors
                    if bval(buffer_vars[t.name]["in_buffer"])
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

        for src, dst, m in edges:
            if bval(m):
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
                base0=ival(buffer_vars[names[0]]["offset"]),
            )
            for names in components.values()
        ]
        return self._justify(units), spilled, chosen_div

    @staticmethod
    def _justify(units: list[_PlacementUnit]) -> dict[str, int]:
        """Slide each placement unit down to the lowest free address. Processing
        in current-base order and giving each the lowest non-conflicting slot
        preserves the relative stacking, so the peak never increases -- it only
        squeezes out the float gaps the search leaves. Returns a name -> address
        map."""
        placed: list[_PlacementUnit] = []
        offsets = {}
        for u in sorted(units, key=lambda u: (u.base0, u.start_time)):
            # lowest base whose [base, base+footprint) clears every already-placed
            # unit that overlaps this one in time
            obstacles = sorted(
                (p.base, p.base + p.footprint)
                for p in placed
                if u.start_time < p.end_time and p.start_time < u.end_time
            )
            base = 0
            for lo, hi in obstacles:
                if base + u.footprint <= lo:
                    break  # fits in the gap below this obstacle
                if base < hi:
                    base = hi  # otherwise bump above it
            u.base = base
            placed.append(u)
            for n in u.members:
                offsets[n] = base
        return offsets

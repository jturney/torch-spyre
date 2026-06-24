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
from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING
import numpy as np


if TYPE_CHECKING:
    import z3
else:
    try:
        import z3

    except ImportError:  # pragma: no cover - exercised only when z3 is absent
        z3 = None

from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivisionBuffer,
    MemoryPlanSolver,
    _assert_in_place_relationships,
)

__all__ = ["CoreDivisionBuffer", "ILPLayoutSolver"]

logger = logging.getLogger(__name__)


@dataclass
class _CoreDivisionBufferWithVars:
    """A :class:`CoreDivisionBuffer` bundled with the z3 variables the solver
    creates for it, so one object flows through the solve instead of a buffer
    list shadowed by a parallel ``name -> {var}`` dict.

    The buffer spans ``[buffer.start_time, buffer.end_time)``; the vars encode
    where (``offset``) and whether (``in_buffer``) it resides in LX, and the
    chosen core division (``division``) with its per-core footprint
    (``eff_size``) and core occupancy (``cores``). ``merge_vars`` maps each
    in-place parent name to the merge bool for that parent->this edge."""

    buffer: CoreDivisionBuffer

    def __post_init__(self):
        self.name = self.buffer.name
        self.end_time = self.buffer.end_time
        self.start_time = self.buffer.start_time
        self.offset = z3.Int(f"off_{self.buffer.name}")
        self.eff_size = z3.Int(f"eff_size_{self.buffer.name}")
        self.in_buffer = z3.Bool(f"in_buffer_{self.buffer.name}")
        self.division = z3.Int(f"div_{self.buffer.name}")
        self.cores = z3.Int(
            f"occ_{self.buffer.name}"
        )  # cores this buffer occupies under chosen div
        self.merge_vars = {
            parent: z3.Bool(f"merge_{parent}_{self.buffer.name}")
            for parent in self.buffer.in_place_parents
        }


@dataclass
class _PlacementUnit:
    """A connected component of in-place-merged buffers placed as one block."""

    members: list[str]
    footprint: int
    start_time: int
    end_time: int
    original_offset: int  # offset z3 chose, before bottom-justify
    justified_offset: int = 0  # final justified offset


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

        # Solve on copies so we never mutate the caller's buffers.
        working = [
            replace(
                b,
                size=int(np.ceil(b.size / self.alignment)),
            )
            for b in buffers
        ]

        offsets, spilled, chosen_div = self._run(working)
        offsets = {k: v * self.alignment for k, v in offsets.items()}

        for b in buffers:
            b.address = None if b.name in spilled else offsets.get(b.name)
            b.chosen_division = chosen_div.get(b.name, b.chosen_division)
        return buffers

    def _run(
        self,
        tensors: list[CoreDivisionBuffer],
    ) -> tuple[dict[str, int], set[str], dict[str, int]]:
        opt = z3.Solver()
        bufs = self._add_buffer_vars(opt, tensors)
        self._add_inplace_relaxation(opt, bufs)
        forced = self._add_core_division(opt, bufs)
        model = self._search(opt, bufs, forced)
        return self._extract(model, bufs)

    @staticmethod
    def _apply_no_overlap_constraint(
        opt: z3.Solver,
        bufs: list[_CoreDivisionBufferWithVars],
        merge_between: dict[tuple[str, str], list[z3.BoolRef]],
    ) -> None:
        def time_overlap(
            a: _CoreDivisionBufferWithVars, b: _CoreDivisionBufferWithVars
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
                opt.add(
                    z3.Implies(
                        z3.And(a.in_buffer, b.in_buffer),
                        z3.Or(
                            a.offset + a.eff_size <= b.offset,
                            b.offset + b.eff_size <= a.offset,
                            *relax,
                        ),
                    )
                )

    def _add_buffer_vars(
        self, opt: z3.Solver, tensors: list[CoreDivisionBuffer]
    ) -> dict[str, _CoreDivisionBufferWithVars]:
        """Allocate per-tensor vars and return ``name ->
        CoreDivisionBufferWithVars``. ``division`` indexes the candidate list,
        ``eff_size`` is the chosen division's per-core footprint (``size`` / its
        ``output_partition``), and ``cores`` its core occupancy. A non-re-divided
        buffer has a single candidate, so ``division`` is pinned to ``0`` and these
        are constants."""
        bufs: dict[str, _CoreDivisionBufferWithVars] = {}
        for t in tensors:
            n = t.name
            bufs[n] = _CoreDivisionBufferWithVars(t)
            offset = bufs[n].offset  # where is the buffer in lx?
            dv = bufs[n].division
            sv = bufs[n].eff_size
            ov = bufs[n].cores  # cores this buffer occupies under chosen div
            opt.add(offset >= 0, offset < self._capacity_units)
            opt.add(dv >= 0, dv <= len(t.core_divisions) - 1)
            per_core = [
                int(np.ceil(t.size / cd.output_partition)) for cd in t.core_divisions
            ]
            partition = [cd.output_partition for cd in t.core_divisions]

            opt.add(
                z3.Or(
                    [
                        z3.And(dv == i, sv == per_core[i], ov == partition[i])
                        for i in range(len(per_core))
                    ]
                )
            )  # tie effective size and core occupancy to the chosen division index
        return bufs

    def _add_inplace_relaxation(
        self,
        opt: z3.Solver,
        bufs: dict[str, _CoreDivisionBufferWithVars],
    ) -> None:
        """In-place reuse as a relaxation of the no-overlap constraint: each
        parent->child edge gets a merge bool that, when active, pins the pair to
        one shared base and lifts their pairwise no-overlap. Chains are induced
        transitively by the shared-offset equalities -- no merge groups, no path
        enumeration -- and ``_assert_in_place_relationships`` guarantees a
        parent/child overlap at exactly one tick, so only directly-edged pairs
        ever need the relaxation. The per-buffer ``merge_vars`` bools are read
        back in ``_extract`` to reconstruct placement units."""
        M = self._capacity_units

        # (src, dst) -> merge bools, consulted by the no-overlap relaxation.
        merge_between: dict[tuple[str, str], list[z3.BoolRef]] = {}
        # A storage slot is handed off linearly, so a buffer reuses at most one
        # parent and is reused by at most one child.
        incoming: dict[str, list[z3.BoolRef]] = {}
        outgoing: dict[str, list[z3.BoolRef]] = {}
        for dst, c in bufs.items():
            for src, edge in c.merge_vars.items():
                src_v, dst_v = bufs[src], bufs[dst]
                # active merge => shared base and both endpoints resident
                opt.add(z3.Implies(edge, src_v.offset == dst_v.offset))
                opt.add(z3.Implies(edge, src_v.in_buffer))
                opt.add(z3.Implies(edge, dst_v.in_buffer))
                # active merge => child reuses the parent's exact per-core storage,
                # so their chosen divisions must have equal per-core footprints.
                opt.add(z3.Implies(edge, dst_v.eff_size == src_v.eff_size))
                # active merge => parent and child must pick slicing-compatible divisions
                self._constrain_merge_division(opt, bufs, src, dst, edge)
                merge_between.setdefault((src, dst), []).append(edge)
                outgoing.setdefault(src, []).append(edge)
                incoming.setdefault(dst, []).append(edge)

        for ms in (*incoming.values(), *outgoing.values()):
            if len(ms) > 1:
                opt.add(z3.Sum(ms) <= 1)

        for sb in bufs.values():
            # if a buffer is resident its top must be below the peak usage.
            opt.add(z3.Implies(sb.in_buffer, sb.offset + sb.eff_size <= M))

        self._apply_no_overlap_constraint(opt, list(bufs.values()), merge_between)

    def _constrain_merge_division(
        self,
        opt: z3.Solver,
        bufs: dict[str, _CoreDivisionBufferWithVars],
        src: str,
        dst: str,
        m: z3.BoolRef,
    ) -> None:
        """Gate an in-place merge on slicing-compatible divisions: when ``m`` is
        active, parent (``src``) and child (``dst``) must pick divisions that
        induce the same per-core slicing of the shared storage. Uses the
        precomputed ``cd_parent_matches`` pairs (see ``_implicate_core_division``);
        no pairs => merge forbidden. A fixed-division endpoint has no ``division``
        to constrain and is already governed by the ``eff_size`` equality.
        """
        pv, cv = bufs[src], bufs[dst]
        child = bufs[dst].buffer
        compatible = child.cd_parent_matches.get(src, [])
        pairs = [z3.And(pv.division == i, cv.division == j) for (i, j) in compatible]
        opt.add(z3.Implies(m, z3.Or(pairs) if pairs else z3.BoolVal(False)))

    def _get_children(
        self, bufs: dict[str, _CoreDivisionBufferWithVars]
    ) -> dict[str, list[tuple[str, list[tuple[int, int]]]]]:
        """parent name -> list of (child name, match_pairs), where ``match_pairs``
        is the child's ``cd_parent_matches[parent]`` (empty when the edge has no
        compatible division). The child's ``parents`` define the edges."""
        children_of: dict[str, list[tuple[str, Any]]] = {}
        for sb in bufs.values():
            t = sb.buffer
            for parent in t.parents:
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
        bufs: dict[str, _CoreDivisionBufferWithVars],
    ) -> set[str]:
        """Pin out of LX the buffers whose non-residency is fixed up front:
        those whose *smallest* candidate footprint still exceeds capacity, and
        those marked ``residency_allowed=False`` (e.g. graph boundaries). This
        set is the search's lower-bound spill seed. Division-dependent residency
        (no consumer, or all consumers mismatch the slicing) is left to
        ``_implicate_core_division`` rather than decided here."""
        forced = set()
        for sb in bufs.values():
            t = sb.buffer
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
                opt.add(z3.Not(sb.in_buffer))
        return forced

    def _implicate_core_division(
        self,
        opt: z3.Solver,
        children_of: dict[str, list[tuple[str, list[tuple[int, int]]]]],
        bufs: dict[str, _CoreDivisionBufferWithVars],
    ) -> None:
        """Slicing-consistency gate: a resident buffer's division must match
        *every* consumer's division under the ``cd_parent_matches`` pairs. The
        buffer carries one ``division``, so requiring all consumers to agree pins
        them to a single per-core slicing; a buffer with diverging consumer
        demands is forced out of LX and re-sliced per consumer on load from main
        memory. An edge with no compatible pair never matches, so a
        partial-reduction buffer resides only when a consumer reads those
        partials under the identical slicing. A divided buffer with no consumer
        edge is gated out the same way -- nothing reads it from LX."""
        for sb in bufs.values():
            t = sb.buffer
            kids = children_of.get(t.name, [])
            if not kids:
                # Nothing consumes this buffer from LX -> it can never reside.
                opt.add(z3.Not(sb.in_buffer))
                continue
            child_matches = []
            for _child, compatible in kids:
                # ``cd_parent_matches`` pairs are the match predicate (physical
                # device-dim view equality, correct across reductions/reshapes
                # where a coeff signature would conflate slicings). Empty => no
                # compatible division => no match, forbidding residency.
                pairs = [
                    z3.And(sb.division == i, bufs[_child].division == j)
                    for (i, j) in compatible
                ]
                child_matches.append(z3.Or(pairs) if pairs else z3.BoolVal(False))
            opt.add(z3.Implies(sb.in_buffer, z3.And(child_matches)))

    def _add_core_division(
        self,
        opt: z3.Solver,
        bufs: dict[str, _CoreDivisionBufferWithVars],
    ) -> set[str]:
        """Wire up children_of, forced spills, and the slicing-match gate.
        Returns the forced-spill set (the search's lower-bound seed). Matching is
        driven entirely by the precomputed ``cd_parent_matches`` pairs."""
        children_of = self._get_children(bufs)
        forced = self._trim_oversized_tensors(opt, bufs)
        self._implicate_core_division(opt, children_of, bufs)
        return forced

    def _search(
        self,
        opt: z3.Solver,
        bufs: dict[str, _CoreDivisionBufferWithVars],
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

        spill_count = z3.Sum([z3.If(sb.in_buffer, 0, 1) for sb in bufs.values()])

        n_tensors = len(bufs)
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
            occ_terms = [z3.If(sb.in_buffer, sb.cores, 0) for sb in bufs.values()]
            if occ_terms:
                occupancy = z3.Sum(occ_terms)
                lo_occ = model.eval(occupancy, model_completion=True).as_long()
                hi_occ = sum(
                    max(cd.output_partition for cd in sb.buffer.core_divisions)
                    for sb in bufs.values()
                    if sb.buffer.core_divisions
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
        self, model: z3.ModelRef, bufs: dict[str, _CoreDivisionBufferWithVars]
    ) -> tuple[dict[str, int], set[str], dict[str, int]]:
        """Read the model into (offsets, spilled, chosen_div). When
        bottom_justify is set, slide each placement unit down to the lowest free
        address (preserving in-place merges, never raising the peak)."""

        def bval(b):
            return z3.is_true(model.eval(b, model_completion=True))

        def ival(x):
            return model.eval(x, model_completion=True).as_long()

        by_name = {name: sb.buffer for name, sb in bufs.items()}
        spilled = {name for name, sb in bufs.items() if not bval(sb.in_buffer)}
        chosen_div = {name: ival(sb.division) for name, sb in bufs.items()}

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
                    name: ival(sb.offset)
                    for name, sb in bufs.items()
                    if bval(sb.in_buffer)
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
                if bval(edge):
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
                original_offset=ival(bufs[names[0]].offset),
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
        for u in sorted(units, key=lambda u: (u.original_offset, u.start_time)):
            # lowest base whose [base, base+footprint) clears every already-placed
            # unit that overlaps this one in time
            obstacles = sorted(
                (p.justified_offset, p.justified_offset + p.footprint)
                for p in placed
                if u.start_time < p.end_time and p.start_time < u.end_time
            )
            base = 0
            for lo, hi in obstacles:
                if base + u.footprint <= lo:
                    break  # fits in the gap below this obstacle
                if base < hi:
                    base = hi  # otherwise bump above it
            u.justified_offset = base
            placed.append(u)
            for n in u.members:
                offsets[n] = base
        return offsets

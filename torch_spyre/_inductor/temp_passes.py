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

# This file contains inductor passes that are only needed as temp fixes

import warnings

import torch
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from .logging_utils import get_inductor_logger
from .constants import SHARED_WEIGHT_UNIT_BMM_CUSTOM_META_KEY
from .pass_utils import copy_fx_custom_meta
from ..ops.fallbacks import FallbackWarning

aten = torch.ops.aten

logger = get_inductor_logger("work_division")

_RESHAPE_OPS = (
    aten.view.default,
    aten.reshape.default,
    aten._unsafe_view.default,
)

mm_to_bmm_pass = PatternMatcherPass(pass_name="unflatten_mm_to_bmm")
bmm_unflatten_pass = PatternMatcherPass(pass_name="unflatten_bmm_batch_dims")


def _is_static_one(value) -> bool:
    try:
        return int(value) == 1
    except (TypeError, ValueError):
        return False


def _is_static_multiple(value, divisor: int) -> bool:
    try:
        return int(value) % divisor == 0
    except (TypeError, ValueError):
        return False


def _has_stick_aligned_matmul_dims(k, n) -> bool:
    return _is_static_multiple(k, 64) and _is_static_multiple(n, 64)


def _static_int(value):
    """Return ``int(value)`` or None if it is symbolic / not convertible."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _node_static_shape(node) -> "list | None":
    """Static shape of ``node``'s fake value, or None if unavailable/symbolic."""
    if not isinstance(node, torch.fx.Node):
        return None
    val = node.meta.get("val")
    if val is None or not hasattr(val, "shape"):
        return None
    return list(val.shape)


# Spyre stick is 128 bytes; elems_per_stick depends on dtype width.
_STICK_BYTES = 128


def _elems_per_stick(node) -> "int | None":
    """Elements per 128-byte stick for ``node``'s dtype, or None if unknown."""
    val = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
    dtype = getattr(val, "dtype", None)
    if dtype is None:
        return None
    try:
        return _STICK_BYTES // torch.empty(0, dtype=dtype).element_size()
    except (TypeError, RuntimeError):
        return None


def _reshape_changes_innermost(node: torch.fx.Node) -> bool:
    """True for a reshape/view that changes the innermost (stick) dimension.

    A reshape that preserves the innermost dim size (e.g. a matmul batch-collapse
    ``[B, M, K] -> [B*M, K]``) keeps every element in the same stick, so it needs
    no data movement. When the innermost size changes, the new element->stick
    mapping produces a stick expression the backend cannot address (rejected by
    ``_check_stick_expr_supported`` in propagate_layouts), so it must be
    materialized off-device. Symbolic shapes return False (leave dynamic alone).
    """
    if node.op != "call_function" or node.target not in _RESHAPE_OPS:
        return False
    if not node.args:
        return False
    in_shape = _node_static_shape(node.args[0])
    out_shape = _node_static_shape(node)
    if not in_shape or not out_shape:
        return False
    in_inner = _static_int(in_shape[-1])
    out_inner = _static_int(out_shape[-1])
    if in_inner is None or out_inner is None:
        return False
    if in_inner == out_inner:
        return False
    # A size-1 innermost is a degenerate squeeze/unsqueeze (e.g. (67,256,1) ->
    # (1,67,256)), not a real stick change -- it is representable and must not be
    # materialized (doing so corrupts the data). Skip it.
    if in_inner <= 1 or out_inner <= 1:
        return False
    # Otherwise representable iff BOTH innermosts are stick-aligned (multiples of
    # elems_per_stick): the reshape stays on stick boundaries, so no cross-stick
    # gather is needed (e.g. a 384 -> 64 split, or a 64 -> 128 merge of full
    # sticks). If either side is a partial stick, the reindex crosses a padded
    # stick boundary and needs a gather -> materialize.
    eps = _elems_per_stick(node)
    if eps is None:
        return False
    return in_inner % eps != 0 or out_inner % eps != 0


def _consumed_on_device(node: torch.fx.Node) -> bool:
    """True if any user of ``node`` is a real op (not just the graph output).

    A reshape whose only use is the graph output is materialized during the
    device->host copy (the host applies the reindex for free), so it does not
    need the CPU round-trip. Only reshapes feeding another on-device op do.
    """
    return any(user.op != "output" for user in node.users)


def _is_materializable_unfold(node: torch.fx.Node) -> bool:
    """True for an ``aten.unfold`` (sliding-window view) that must be materialized.

    ``aten.unfold`` turns one input dim into two output dims -- num_slices (stride
    ``parent_stride * step``) and window (stride ``parent_stride``) -- that index
    the same parent storage. An on-device read then folds both loop vars into one
    physical coordinate (see ``compute_coordinates`` / ``_check_stick_expr_supported``),
    which either mis-addresses (wrong values) or raises an unsupported multi-var
    stick expression. Materializing to a compact buffer fixes both.

    v1 predicate (correctness-first): materialize any non-degenerate unfold
    (window ``size > 1``). A size-1 window is a no-op view that addresses fine and
    must NOT be materialized. This is intentionally broad -- it also covers
    non-overlapping-but-stick-colliding cases (e.g. a 1-D ``unfold(0,4,4)`` whose
    slice dim still lands in the stick coordinate as ``4*d0 + d1``) -- and only
    ever over-materializes with a copy cost, never affecting correctness.
    """
    if node.op != "call_function" or node.target != aten.unfold.default:
        return False
    if len(node.args) < 4:
        return False
    size = _static_int(node.args[2])
    if size is None:
        return False
    return size > 1


def reroute_overlapping_unfold(graph: torch.fx.Graph) -> None:
    """Route materializable ``aten.unfold`` views through the CPU round-trip.

    An unfold view read by an on-device op mis-addresses (its two output dims
    index the same parent storage; the coordinate computation folds both loop
    vars into one physical coordinate). Rewrite such an unfold to
    ``spyre.unfold_via_cpu`` so the consumer receives a compact, non-overlapping
    tensor. Correctness fallback with a device<->host copy cost; each rewrite
    emits a ``FallbackWarning``. Mirrors ``reroute_stick_misaligned_reshapes``.
    """
    for node in list(graph.nodes):
        if not _is_materializable_unfold(node):
            continue
        if not _consumed_on_device(node):
            continue
        inp = node.args[0]
        dimension, size, step = (
            int(node.args[1]),
            int(node.args[2]),
            int(node.args[3]),
        )
        with graph.inserting_after(node):
            new_node = graph.call_function(
                torch.ops.spyre.unfold_via_cpu.default,
                (inp, dimension, size, step),
            )
        new_node.meta.update(node.meta)
        copy_fx_custom_meta(node, new_node)
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        warnings.warn(
            f"unfold view (dim={dimension}, size={size}, step={step}) routed "
            "through a CPU round-trip (spyre.unfold_via_cpu): correctness "
            "fallback with a device<->host copy cost",
            category=FallbackWarning,
        )
    graph.lint()


def reroute_stick_misaligned_reshapes(graph: torch.fx.Graph) -> None:
    """Route stick-misaligned reshapes through the CPU round-trip fallback.

    A reshape/view that changes the innermost (stick) dimension and is then
    consumed by an on-device op would fail in propagate_layouts with an
    unsupported stick expression. Rewrite it to ``spyre.reshape_via_cpu`` so the
    consumer receives a stick-aligned tensor. This is a correctness fallback with
    a device<->host copy cost, so each rewrite emits a ``FallbackWarning`` naming
    the shapes. Runs after the mm/bmm passes, so matmul batch-reshapes (which
    preserve the innermost dim and so are never matched here) are untouched.
    """
    for node in list(graph.nodes):
        if not _reshape_changes_innermost(node):
            continue
        if not _consumed_on_device(node):
            continue
        inp = node.args[0]
        out_shape_raw = _node_static_shape(node)
        in_shape = _node_static_shape(inp)
        if out_shape_raw is None or in_shape is None:
            continue  # guarded by _reshape_changes_innermost, but narrow for mypy
        out_shape = [int(s) for s in out_shape_raw]
        with graph.inserting_after(node):
            new_node = graph.call_function(
                torch.ops.spyre.reshape_via_cpu.default, (inp, out_shape)
            )
        new_node.meta.update(node.meta)
        copy_fx_custom_meta(node, new_node)
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        warnings.warn(
            f"stick-misaligned reshape {tuple(in_shape)} -> {tuple(out_shape)} "
            "routed through a CPU round-trip (spyre.reshape_via_cpu): correctness "
            "fallback with a device<->host copy cost",
            category=FallbackWarning,
        )
    graph.lint()


def _node_shape(node: torch.fx.Node) -> list[int] | None:
    val = node.meta.get("val")
    shape = getattr(val, "shape", None)
    if shape is None:
        return None
    return list(shape)


def _mark_static_unit_batch_bmm(
    bmm_node: torch.fx.Node, lhs_node: torch.fx.Node, rhs_node: torch.fx.Node
) -> None:
    lhs_shape = _node_shape(lhs_node)
    rhs_shape = _node_shape(rhs_node)
    out_shape = _node_shape(bmm_node)
    if lhs_shape is None or rhs_shape is None or out_shape is None:
        return
    if len(lhs_shape) != 3 or len(rhs_shape) != 3 or len(out_shape) != 3:
        return
    if not (
        _is_static_one(lhs_shape[0])
        and _is_static_one(rhs_shape[0])
        and _is_static_one(out_shape[0])
    ):
        return
    if not (
        lhs_shape[1] == out_shape[1]
        and lhs_shape[2] == rhs_shape[1]
        and rhs_shape[2] == out_shape[2]
    ):
        return
    if not _has_stick_aligned_matmul_dims(lhs_shape[2], rhs_shape[2]):
        return
    custom = dict(bmm_node.meta.get("custom") or {})
    custom[SHARED_WEIGHT_UNIT_BMM_CUSTOM_META_KEY] = {"batch_dim": 0}
    bmm_node.meta["custom"] = custom


def _is_direct_unit_bmm_operand(node: torch.fx.Node) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op in ("placeholder", "get_attr"):
        return True
    if node.op == "call_function" and node.target == aten.expand.default:
        base = node.args[0]
        return isinstance(base, torch.fx.Node) and base.op in (
            "placeholder",
            "get_attr",
        )
    return False


def _mark_direct_static_unit_batch_bmm(
    bmm_node: torch.fx.Node, lhs_node: torch.fx.Node, rhs_node: torch.fx.Node
) -> None:
    """Mark direct rank-3 B=1 BMMs without catching unflattened attention views."""
    if not _is_direct_unit_bmm_operand(rhs_node):
        return

    for arg in (lhs_node, rhs_node):
        if (
            isinstance(arg, torch.fx.Node)
            and arg.op == "call_function"
            and arg.target in _RESHAPE_OPS
        ):
            return

    bmm_users = list(bmm_node.users.keys())
    if len(bmm_users) == 1:
        output_view = bmm_users[0]
        if (
            isinstance(output_view, torch.fx.Node)
            and output_view.op == "call_function"
            and output_view.target in _RESHAPE_OPS
        ):
            output_shape = output_view.args[1]
            if isinstance(output_shape, (list, tuple)) and len(output_shape) > 3:
                return

    _mark_static_unit_batch_bmm(bmm_node, lhs_node, rhs_node)


def mark_direct_unit_bmm_pass(graph: torch.fx.Graph) -> None:
    for node in graph.nodes:
        if node.op != "call_function" or node.target != aten.bmm.default:
            continue
        if len(node.args) != 2:
            continue
        lhs_node, rhs_node = node.args
        _mark_direct_static_unit_batch_bmm(node, lhs_node, rhs_node)


@register_graph_pattern(
    CallFunction(aten.mm.default, Arg(), Arg()),
    pass_dict=mm_to_bmm_pass,
)
def _unflatten_mm_to_bmm(
    match: Match, mat1_node: torch.fx.Node, mat2_node: torch.fx.Node
) -> None:
    """
    Convert view(3D→2D) → mm(2D, 2D) → view(2D→3D) into bmm(3D, unsqueeze(2D)).

    When torch.matmul is called with a batched input and a 2D weight, the
    decomposition flattens the batch dimensions:
      1. view(input, [B*M, K])
      2. mm(flattened, weight) -> [B*M, N]
      3. view(mm_result, [B, M, N])

    The Spyre backend handles bmm better. This pass converts the pattern
    into a semantically correct bmm by unsqueezeing and expanding the 2D
    weight to match the batch dimension of the input.
    """
    node = match.nodes[-1]
    graph = node.graph
    lhs, rhs = mat1_node, mat2_node

    # LHS must be a reshape that flattens a higher-dim tensor to 2D
    if not (
        isinstance(lhs, torch.fx.Node)
        and lhs.op == "call_function"
        and lhs.target in _RESHAPE_OPS
    ):
        return
    lhs_input = lhs.args[0]
    if not (isinstance(lhs_input, torch.fx.Node) and "val" in lhs_input.meta):
        return
    lhs_orig_shape = list(lhs_input.meta["val"].shape)

    # RHS must be a plain 2D tensor (not a reshaped one)
    if not (isinstance(rhs, torch.fx.Node) and "val" in rhs.meta):
        return
    rhs_shape = list(rhs.meta["val"].shape)
    if len(rhs_shape) != 2:
        return

    # The mm result must feed into exactly one view that restores batch dims
    mm_users = list(node.users.keys())
    if len(mm_users) != 1:
        return
    output_view = mm_users[0]
    if not (output_view.op == "call_function" and output_view.target in _RESHAPE_OPS):
        return
    output_shape = output_view.args[1]
    if not isinstance(output_shape, (list, tuple)):
        return
    if len(output_shape) <= 2:
        return

    # Verify the output shape's batch dims match the original input's
    if list(output_shape[:-1]) != lhs_orig_shape[:-1]:
        return

    # Build the bmm: bmm(lhs_orig, unsqueeze(rhs, 0).expand(B, K, N))
    batch_dims = lhs_orig_shape[:-2]  # e.g. [2] from [2, 64, 128]
    K, N = rhs_shape

    with graph.inserting_before(node):
        # unsqueeze weight to 3D+: [K, N] → [1, ..., 1, K, N]
        unsqueezed = rhs
        rhs_dtype = rhs.meta["val"].dtype
        unsqueezed_shape = list(rhs_shape)
        for i in range(len(batch_dims)):
            unsqueezed = graph.call_function(
                aten.unsqueeze.default,
                args=(unsqueezed, 0),
            )
            unsqueezed_shape = [1] + unsqueezed_shape
            unsqueezed.meta["val"] = torch.empty(
                unsqueezed_shape, dtype=rhs_dtype, device="meta"
            )

        # expand to match batch dims: [1, ..., 1, K, N] → [B, ..., K, N]
        expanded_shape = batch_dims + [K, N]
        expanded = graph.call_function(
            aten.expand.default,
            args=(unsqueezed, expanded_shape),
        )
        expanded.meta["val"] = torch.empty(
            expanded_shape, dtype=rhs_dtype, device="meta"
        )

        # Use spyre.batched_matmul for >3D to avoid FakeTensorUpdater crash
        # (aten.bmm requires exactly 3D inputs)
        target = (
            torch.ops.spyre.batched_matmul.default
            if len(output_shape) > 3
            else aten.bmm.default
        )
        bmm_node = graph.call_function(
            target,
            args=(lhs_input, expanded),
        )
        bmm_node.meta["val"] = torch.empty(output_shape, dtype=rhs_dtype, device="meta")
        copy_fx_custom_meta(node, bmm_node)
        _mark_static_unit_batch_bmm(bmm_node, lhs_input, expanded)

    # Replace all uses of mm and output view with the bmm
    node.replace_all_uses_with(bmm_node)
    output_view.replace_all_uses_with(bmm_node)

    # Clean up dead nodes
    graph.erase_node(output_view)
    graph.erase_node(node)
    if not lhs.users:
        graph.erase_node(lhs)


def _is_batch_collapsing_reshape(node: torch.fx.Node) -> bool:
    """Check if a node is a reshape that collapses batch dims into a single dim."""
    if not isinstance(node, torch.fx.Node):
        return False
    if node.op != "call_function":
        return False
    if node.target not in _RESHAPE_OPS:
        return False
    # The reshape output should be 3D (batch_product, M, K)
    output_shape = node.args[1]
    if not isinstance(output_shape, (list, tuple)) or len(output_shape) != 3:
        return False
    # The input should be higher dimensional
    input_node = node.args[0]
    if isinstance(input_node, torch.fx.Node) and "val" in input_node.meta:
        input_ndim = input_node.meta["val"].dim()
        return input_ndim > 3
    return False


@register_graph_pattern(
    CallFunction(aten.bmm.default, Arg(), Arg()),
    pass_dict=bmm_unflatten_pass,
)
def _unflatten_bmm_batch_dims(
    match: Match, mat1_node: torch.fx.Node, mat2_node: torch.fx.Node
) -> None:
    """
    Undo the matmul decomposition's flattening of batch dimensions into 3D bmm.

    The matmul decomposition in torch/_decomp/decompositions.py converts N-D
    matmuls (e.g. 4D SDPA attention) into 3D by:
      1. expand(input, [B, H, M, K]) -> reshape([B*H, M, K])
      2. expand(input, [B, H, K, N]) -> reshape([B*H, K, N])
      3. bmm(reshaped1, reshaped2) -> [B*H, M, N]
      4. view(bmm_result, [B, H, M, N]) -> back to original dims

    This pass removes the reshape/view wrapper so the bmm operates on the
    original higher-dimensional tensors, which the Spyre backend can handle
    natively via its 4D batch matmul lowering.

    This is needed as the flattened views are not supported by the current
    backend. When KTIR is implemented this pass can be dropped.
    """
    node = match.nodes[-1]
    graph = node.graph
    lhs_reshape, rhs_reshape = mat1_node, mat2_node

    # Both inputs must be reshape/view that collapse batch dims to 3D
    if not _is_batch_collapsing_reshape(lhs_reshape):
        return
    if not _is_batch_collapsing_reshape(rhs_reshape):
        return

    # The bmm result must feed into exactly one view that restores the batch dims
    bmm_users = list(node.users.keys())
    if len(bmm_users) != 1:
        return
    output_view = bmm_users[0]
    if not (output_view.op == "call_function" and output_view.target in _RESHAPE_OPS):
        return

    output_shape = output_view.args[1]
    if len(output_shape) <= 3:
        return

    # Get the original (pre-reshape) tensors
    lhs_orig = lhs_reshape.args[0]  # the expand or original tensor
    rhs_orig = rhs_reshape.args[0]

    # Replace the 3D bmm with a spyre.batched_matmul that accepts N-D inputs.
    # Using aten.bmm.default with >3D args would crash FakeTensorUpdater.
    with graph.inserting_before(node):
        matmul_node = graph.call_function(
            torch.ops.spyre.batched_matmul.default,
            args=(lhs_orig, rhs_orig),
        )
        matmul_node.meta["val"] = output_view.meta["val"]
        copy_fx_custom_meta(node, matmul_node)

    # Replace all uses of the output view with the new matmul
    output_view.replace_all_uses_with(matmul_node)
    node.replace_all_uses_with(matmul_node)
    graph.erase_node(output_view)
    graph.erase_node(node)

    # Clean up dead reshape nodes
    for reshape_node in (lhs_reshape, rhs_reshape):
        if not reshape_node.users:
            expand_node = reshape_node.args[0]
            graph.erase_node(reshape_node)
            # Also remove the expand if it's now unused
            if (
                isinstance(expand_node, torch.fx.Node)
                and expand_node.op == "call_function"
                and expand_node.target == aten.expand.default
                and not expand_node.users
            ):
                graph.erase_node(expand_node)

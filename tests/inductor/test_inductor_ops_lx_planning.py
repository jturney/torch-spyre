import functools
import torch_spyre
from torch_spyre._inductor import config
import os
import sys
import torch
from torch.utils import _pytree as pytree

from torch._dynamo.testing import make_test_cls_with_patches

import unittest
from utils_inductor import compare_with_cpu, copy_tests, TestFailure

_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(_test_dir)

import inductor.test_inductor_ops  # noqa: E402


# xfail by default, set is_skip=True to skip
test_failures = {
    "test_cat_1d_dim0": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_1d_dim0_three_tensors": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_2d_dim0_diff_size": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_2d_dim0_three_tensors": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_2d_dim1_diff_size": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_3d_dim0": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_3d_dim1": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_3d_dim1_size1": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_3d_dim2": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_4d_dim0": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_4d_dim1": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_4d_dim2": TestFailure(("lx_planning"), is_skip=True),
    "test_cat_4d_dim3": TestFailure(("lx_planning"), is_skip=True),
    "test_activation_fn_mish_fp16": TestFailure(("lx_planning"), is_skip=True),
    "test_activation_fn_silu_fp16": TestFailure(("lx_planning"), is_skip=True),
    "test_addmm_out_basic": TestFailure(("lx_planning"), is_skip=True),
}


def make_lx_planning_class(cls):
    return make_test_cls_with_patches(
        cls,
        "LxPlanning",
        "",
        (torch_spyre._inductor.config, "lx_planning", True),
        (torch_spyre._inductor.config, "allow_all_ops_in_lx_planning", True),
        (torch_spyre._inductor.config, "sencores", 1),
    )


POINTWISE_TEST_FAILURES = {
    "test_add_broadcast_cpu_256_67x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_addmm_1152_10x1152_1152x1152": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_addmm_out_basic": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_addmm_scaled_alpha_0_5": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_alias_operands_cpu_pow_67x71x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_alias_operands_cube_256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_alias_operands_cube_67x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_alias_operands_cube_67x71x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_alias_operands_double_67x71x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_alias_operands_square_67x71x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_alias_operands_triple_256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_alias_operands_triple_67x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_alias_operands_triple_67x71x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_attention_3d_batch_size_1": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_attention_3d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_attention_4d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_bmm_bmm_2x256x1_2x1x128": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_bmm_bmm_2x55x2_2x2x99": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_bmm_bmm_2x99x65_2x65x55": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_bmm_bmm_3x17x256_3x256x128": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_bmm_bmm_3x1x256_3x256x128": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_cat_1d_dim0": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cat_1d_dim0_three_tensors": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_cat_2d_dim0_diff_size": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cat_2d_dim0_three_tensors": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_cat_2d_dim1_diff_size": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cat_3d_dim0": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cat_3d_dim1": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cat_3d_dim1_size1": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cat_3d_dim2": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cat_4d_dim0": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cat_4d_dim1": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cat_4d_dim2": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cat_4d_dim3_fp32": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cat_4d_dim3": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_clone_bool_1d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_clone_bool_2d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_clone_bool_3d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_eq_1d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_eq_2d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_eq_3d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_eq_broadcast": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_ge_1d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_ge_2d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_ge_3d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_ge_broadcast": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_gt_1d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_gt_2d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_gt_3d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_gt_broadcast": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_le_1d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_le_2d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_le_3d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_le_broadcast": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_lt_1d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_lt_2d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_lt_3d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_lt_broadcast": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_ne_1d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_ne_2d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_ne_3d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_cmp_ne_broadcast": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_fallback_1d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_fallback_2d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_fallback_3d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_full_value_1": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_full_value_2": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_inplace_copy_copy_bool": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_inplace_op_add_1d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_inplace_op_add_2d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_inplace_op_add_3d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_inplace_op_mul_1d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_inplace_op_mul_2d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_inplace_op_mul_3d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_isin_out_tensor_tensor": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_isin_tensor_tensor": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_item_from_computation": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_layernorm_2d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_linear_2d_bias": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_linear_2d_no_bias": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_linear_3d_bias": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_linear_3d_no_bias": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_logical_not_logical_not_1d_bool": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_logical_not_logical_not_1d_fp16": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_logical_not_logical_not_2d_bool": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_logical_not_logical_not_2d_fp16": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_logical_not_logical_not_3d_bool": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_logical_not_logical_not_3d_fp16": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_logical_not_logical_not_4d_bool": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_logical_not_logical_not_4d_fp16": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_logical_not_logical_not_bool_single_elem": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_logical_not_logical_not_fp16_single_elem": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_matmul_matmul_2x3x55x2_2x3x2x99": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_matmul_matmul_2x3x99x1_2x3x1x55": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_matmul_matmul_2x3x99x65_2x3x65x55": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_matmul_matmul_2x55x2_2x2x99": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_matmul_matmul_2x64x128_128x16384": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_matmul_matmul_2x99x1_1x55": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_matmul_matmul_2x99x1_2x1x55": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_matmul_matmul_2x99x65_2x65x55": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_matmul_matmul_3x17x256_3x256x128": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_matmul_matmul_3x18x128x256_3x18x256x128": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_matmul_matmul_3x1x256_3x256x128": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_matmul_matmul_512x256_256x128": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_matmul_matmul_55x2_2x99": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_matmul_matmul_99x1_1x55": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_matmul_matmul_99x65_65x55": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_keepdim0_sum_2d_dim_0": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_keepdim0_sum_2d_dim_1": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_keepdim0_sum_3d_dim_1": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_keepdim0_sum_3d_dim_2": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_keepdim0_sum_4d_dim_0": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_keepdim0_sum_4d_dim_1": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_keepdim0_sum_4d_dim_2": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_keepdim0_sum_4d_dim_3": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_keepdim1_sum_2d_dim_0": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_keepdim1_sum_2d_dim_1": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_keepdim1_sum_3d_dim_0": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_keepdim1_sum_3d_dim_1": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_keepdim1_sum_3d_dim_2": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_keepdim1_sum_4d_dim_0": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_keepdim1_sum_4d_dim_1": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_keepdim1_sum_4d_dim_2": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_keepdim1_sum_4d_dim_3": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_sub_broadcast_2d_dim_0": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_sub_broadcast_2d_dim_1": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_sub_broadcast_4d_dim_0": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_sub_broadcast_4d_dim_1": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_sub_broadcast_4d_dim_2": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_max_sub_broadcast_4d_dim_3": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_mm_mm_55x2_2x99": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_mm_mm_67x255_255x128": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_mm_mm_67x256_256x128": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_mm_mm_67x67_67x67": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_pad_2d_both_dims": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_pad_2d_dim0_left": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_pad_2d_dim0_left_only": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_pad_2d_last_dim_left_and_right_stick_aligned": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pad_2d_last_dim_left_stick_aligned": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pad_2d_last_dim_left_two_sticks": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pad_2d_last_dim_right": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_pad_3d_dim0_left": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_pad_3d_dim1_left": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_pad_3d_dim1_right": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_pad_3d_last_dim_right": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_pad_4d_dim0_left": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_permute_2d_1_0": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_permute_3d_0_2_1": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_permute_4d_0_2_1_3": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_permute_4d_0_3_1_2": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_permute_4d_0_m2_m1_1": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_permute_5d_0_2_3_4_1": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_pointwise_binary_op_add_256_256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_binary_op_add_67x256_67x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_binary_op_add_67x71x256_67x71x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_binary_op_add_7x12x32x64_7x12x32x64": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_binary_op_div_256_256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_binary_op_div_67x256_67x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_binary_op_div_67x71x256_67x71x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_binary_op_div_7x12x32x64_7x12x32x64": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_binary_op_fp32_add_fp32": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_binary_op_fp32_div_fp32": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_binary_op_fp32_mul_fp32": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_binary_op_fp32_sub_fp32": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_binary_op_mul_256_256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_binary_op_mul_67x256_67x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_binary_op_mul_67x71x256_67x71x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_binary_op_mul_7x12x32x64_7x12x32x64": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_binary_op_sub_256_256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_binary_op_sub_67x256_67x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_binary_op_sub_67x71x256_67x71x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_binary_op_sub_7x12x32x64_7x12x32x64": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_unary_op_abs_67x71x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_unary_op_exp_67x71x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_unary_op_neg_67x71x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_unary_op_reciprocal_67x71x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_unary_op_relu_67x71x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_pointwise_unary_op_tanh_67x71x256": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_rmsnorm_2d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_rmsnorm_3d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_rmsnorm_4d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_scalar_cpu_add_1d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_scalar_cpu_add_2d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_scalar_cpu_add_3d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_scalar_cpu_add_4d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_scalar_cpu_combined_1d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_scalar_cpu_combined_2d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_scalar_cpu_combined_3d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_scalar_cpu_combined_4d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_scalar_cpu_div_1d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_scalar_cpu_div_2d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_scalar_cpu_div_3d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_scalar_cpu_div_4d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_scalar_cpu_mul_1d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_scalar_cpu_mul_2d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_scalar_cpu_mul_3d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_scalar_cpu_mul_4d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_scalar_cpu_sub_1d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_scalar_cpu_sub_2d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_scalar_cpu_sub_3d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_scalar_cpu_sub_4d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_scalar_cpu_true_divide_1d": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_scalar_cpu_true_divide_2d": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_scalar_cpu_true_divide_3d": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_scalar_cpu_true_divide_4d": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_sdpa_mha_prefill_causal": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_sdpa_mha_prefill": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_sdpa_mha_prefill_mask": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_softmax_softmax_2d_dim0": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_softmax_softmax_2d_dim1": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_softmax_softmax_3d_dim0": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_softmax_softmax_3d_dim1": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_softmax_softmax_3d_dim2": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_softplus_3d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_softplus_4d": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_squeeze_reduction_sum_3d0": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_squeeze_reduction_sum_3d1": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_squeeze_reduction_sum_4d0": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_squeeze_reduction_sum_4d1": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_squeeze_reduction_sum_4d2": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_squeeze_single_2d1": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_squeeze_single_3d0": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_squeeze_single_3d1": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_squeeze_single_3d2": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_squeeze_single_4d0": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_squeeze_single_4d1": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_squeeze_single_4d2": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_squeeze_single_4d3": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_sum_keepdim0_sum_2d_dim_0": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_sum_keepdim0_sum_3d_dim_1": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_sum_keepdim0_sum_3d_dim_2": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_sum_keepdim1_sum_2d_dim_0": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_sum_keepdim1_sum_3d_dim_1": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_sum_keepdim1_sum_3d_dim_2": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_t_2d_1088x320": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_t_2d_320x320": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_t_2d_contiguous_4096x49280": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_t_2d_contiguous_49280x4096": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_transpose_2d_contiguous_dim_0_1": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_transpose_2d_contiguous_dim_0_2": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_transpose_2d_contiguous_dim_0_2_same_dim": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_transpose_2d_contiguous_dim_1_2": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_transpose_2d_dim_0_1": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_transpose_2d_dim_0_2": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_transpose_2d_dim_0_2_same_dim": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_transpose_2d_dim_1_2": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_transpose_3d_contiguous_dim_0_1": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_transpose_3d_contiguous_dim_0_2": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_transpose_3d_contiguous_dim_0_2_same_dim": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_transpose_3d_contiguous_dim_1_2": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_transpose_3d_dim_0_1": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_transpose_3d_dim_0_2": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_transpose_3d_dim_0_2_same_dim": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_transpose_3d_dim_1_2": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_transpose_4d_contiguous_dim_0_1": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_transpose_4d_contiguous_dim_0_3": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_transpose_4d_contiguous_dim_1_2": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_transpose_4d_contiguous_dim_1_3": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_transpose_4d_contiguous_dim_2_3": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_transpose_4d_dim_0_1": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_transpose_4d_dim_0_3": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_transpose_4d_dim_1_2": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_transpose_4d_dim_1_3": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_transpose_4d_dim_2_3": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_unsqueeze_broadcast_add_1d0": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_unsqueeze_broadcast_add_2d0": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_unsqueeze_broadcast_add_2d1": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_unsqueeze_broadcast_add_3d0": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_unsqueeze_broadcast_add_3d1": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_unsqueeze_broadcast_add_3d2": TestFailure(
        ("lx_planning_pointwise"), is_skip=True
    ),
    "test_unsqueeze_single_1d0": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_unsqueeze_single_2d0": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_unsqueeze_single_2d1": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_unsqueeze_single_3d0": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_unsqueeze_single_3d1": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_unsqueeze_single_3d2": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_unsqueeze_single_4d0": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_unsqueeze_single_4d1": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_unsqueeze_single_4d2": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_unsqueeze_single_4d3": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_where_eq_1d256": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_where_ge_1d256": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_where_gt_1d256": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_where_le_1d256": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_where_lt_1d256": TestFailure(("lx_planning_pointwise"), is_skip=True),
    "test_where_ne_1d256": TestFailure(("lx_planning_pointwise"), is_skip=True),
}


class LxPlanningTwoOpPointwiseAdditionTest(unittest.TestCase):
    def wrap_pointwise(self, fn):
        @functools.wraps(fn)
        def make_seq_of_ops(*fn_args, **fn_kwargs):
            result = fn(*fn_args, **fn_kwargs)
            return pytree.tree_map(
                lambda x: x + x if isinstance(x, torch.Tensor) else x, result
            )

        return make_seq_of_ops

    def compare_with_cpu(self, fn, *args, **kwargs):
        kwargs["cpu_compile"] = False
        return compare_with_cpu(self.wrap_pointwise(fn), *args, **kwargs)

    def compare(
        self,
        fn,
        *args,
        atol=0.0,
        rtol=0.0,
        cpu_atol=0.1,
        cpu_rtol=0.1,
        needs_device=False,
    ):
        return compare_with_cpu(
            self.wrap_pointwise(fn),
            *args,
            atol=cpu_atol,
            rtol=cpu_rtol,
            needs_device=needs_device,
            cpu_compile=False,
        )


copy_tests(
    make_lx_planning_class(inductor.test_inductor_ops.TestOps),
    LxPlanningTwoOpPointwiseAdditionTest,
    "lx_planning_pointwise",
    POINTWISE_TEST_FAILURES if not config.tests_lx_planning_run_skips else None,
)


POINTWISE_SUBTRACTION_TEST_FAILURES = {
    "test_add_broadcast_cpu_256_67x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_addmm_1152_10x1152_1152x1152": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_addmm_out_basic": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_addmm_scaled_alpha_0_5": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_alias_operands_cpu_pow_67x71x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_alias_operands_cube_256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_alias_operands_cube_67x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_alias_operands_cube_67x71x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_alias_operands_double_67x71x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_alias_operands_square_67x71x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_alias_operands_triple_256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_alias_operands_triple_67x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_alias_operands_triple_67x71x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_attention_3d_batch_size_1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_attention_3d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_attention_4d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_bmm_bmm_2x256x1_2x1x128": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_bmm_bmm_2x55x2_2x2x99": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_bmm_bmm_2x99x65_2x65x55": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_bmm_bmm_3x17x256_3x256x128": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_bmm_bmm_3x1x256_3x256x128": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_cat_1d_dim0": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_cat_1d_dim0_three_tensors": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_cat_2d_dim0_diff_size": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_cat_2d_dim0_three_tensors": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_cat_2d_dim1_diff_size": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_cat_3d_dim0": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_cat_3d_dim1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_cat_3d_dim1_size1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_cat_3d_dim2": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_cat_4d_dim0": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_cat_4d_dim1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_cat_4d_dim2": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_cat_4d_dim3_fp32": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_cat_4d_dim3": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_clone_bool_1d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_clone_bool_2d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_clone_bool_3d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_cmp_eq_1d": TestFailure(("lx_planning_pointwise_subtraction"), is_skip=True),
    "test_cmp_eq_2d": TestFailure(("lx_planning_pointwise_subtraction"), is_skip=True),
    "test_cmp_eq_3d": TestFailure(("lx_planning_pointwise_subtraction"), is_skip=True),
    "test_cmp_eq_broadcast": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_cmp_ge_1d": TestFailure(("lx_planning_pointwise_subtraction"), is_skip=True),
    "test_cmp_ge_2d": TestFailure(("lx_planning_pointwise_subtraction"), is_skip=True),
    "test_cmp_ge_3d": TestFailure(("lx_planning_pointwise_subtraction"), is_skip=True),
    "test_cmp_ge_broadcast": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_cmp_gt_1d": TestFailure(("lx_planning_pointwise_subtraction"), is_skip=True),
    "test_cmp_gt_2d": TestFailure(("lx_planning_pointwise_subtraction"), is_skip=True),
    "test_cmp_gt_3d": TestFailure(("lx_planning_pointwise_subtraction"), is_skip=True),
    "test_cmp_gt_broadcast": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_cmp_le_1d": TestFailure(("lx_planning_pointwise_subtraction"), is_skip=True),
    "test_cmp_le_2d": TestFailure(("lx_planning_pointwise_subtraction"), is_skip=True),
    "test_cmp_le_3d": TestFailure(("lx_planning_pointwise_subtraction"), is_skip=True),
    "test_cmp_le_broadcast": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_cmp_lt_1d": TestFailure(("lx_planning_pointwise_subtraction"), is_skip=True),
    "test_cmp_lt_2d": TestFailure(("lx_planning_pointwise_subtraction"), is_skip=True),
    "test_cmp_lt_3d": TestFailure(("lx_planning_pointwise_subtraction"), is_skip=True),
    "test_cmp_lt_broadcast": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_cmp_ne_1d": TestFailure(("lx_planning_pointwise_subtraction"), is_skip=True),
    "test_cmp_ne_2d": TestFailure(("lx_planning_pointwise_subtraction"), is_skip=True),
    "test_cmp_ne_3d": TestFailure(("lx_planning_pointwise_subtraction"), is_skip=True),
    "test_cmp_ne_broadcast": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_fallback_1d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_fallback_2d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_fallback_3d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_inplace_copy_copy_bool": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_inplace_op_add_1d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_inplace_op_add_2d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_inplace_op_add_3d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_inplace_op_mul_1d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_inplace_op_mul_2d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_inplace_op_mul_3d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_isin_out_tensor_tensor": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_isin_tensor_tensor": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_item_from_computation": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_layernorm_2d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_linear_2d_bias": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_linear_2d_no_bias": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_linear_3d_bias": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_linear_3d_no_bias": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_logical_not_logical_not_1d_bool": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_logical_not_logical_not_1d_fp16": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_logical_not_logical_not_2d_bool": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_logical_not_logical_not_2d_fp16": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_logical_not_logical_not_3d_bool": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_logical_not_logical_not_3d_fp16": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_logical_not_logical_not_4d_bool": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_logical_not_logical_not_4d_fp16": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_logical_not_logical_not_bool_single_elem": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_logical_not_logical_not_fp16_single_elem": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_matmul_matmul_2x3x55x2_2x3x2x99": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_matmul_matmul_2x3x99x1_2x3x1x55": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_matmul_matmul_2x3x99x65_2x3x65x55": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_matmul_matmul_2x55x2_2x2x99": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_matmul_matmul_2x64x128_128x16384": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_matmul_matmul_2x99x1_1x55": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_matmul_matmul_2x99x1_2x1x55": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_matmul_matmul_2x99x65_2x65x55": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_matmul_matmul_3x17x256_3x256x128": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_matmul_matmul_3x18x128x256_3x18x256x128": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_matmul_matmul_3x1x256_3x256x128": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_matmul_matmul_512x256_256x128": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_matmul_matmul_55x2_2x99": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_matmul_matmul_99x1_1x55": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_matmul_matmul_99x65_65x55": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_keepdim0_sum_2d_dim_0": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_keepdim0_sum_2d_dim_1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_keepdim0_sum_3d_dim_1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_keepdim0_sum_3d_dim_2": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_keepdim0_sum_4d_dim_0": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_keepdim0_sum_4d_dim_1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_keepdim0_sum_4d_dim_2": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_keepdim0_sum_4d_dim_3": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_keepdim1_sum_2d_dim_0": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_keepdim1_sum_2d_dim_1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_keepdim1_sum_3d_dim_0": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_keepdim1_sum_3d_dim_1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_keepdim1_sum_3d_dim_2": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_keepdim1_sum_4d_dim_0": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_keepdim1_sum_4d_dim_1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_keepdim1_sum_4d_dim_2": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_keepdim1_sum_4d_dim_3": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_sub_broadcast_2d_dim_0": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_sub_broadcast_2d_dim_1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_sub_broadcast_4d_dim_0": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_sub_broadcast_4d_dim_1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_sub_broadcast_4d_dim_2": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_max_sub_broadcast_4d_dim_3": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_mm_mm_55x2_2x99": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_mm_mm_67x255_255x128": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_mm_mm_67x256_256x128": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_mm_mm_67x67_67x67": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pad_2d_both_dims": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pad_2d_dim0_left": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pad_2d_dim0_left_only": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pad_2d_last_dim_left_and_right_stick_aligned": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pad_2d_last_dim_left_stick_aligned": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pad_2d_last_dim_left_two_sticks": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pad_2d_last_dim_right": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pad_3d_dim0_left": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pad_3d_dim1_left": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pad_3d_dim1_right": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pad_3d_last_dim_right": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pad_4d_dim0_left": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_permute_4d_0_2_1_3": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_permute_4d_0_3_1_2": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_softmax_softmax_2d_dim0": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_softmax_softmax_2d_dim1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_softmax_softmax_3d_dim0": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_softmax_softmax_3d_dim1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_softmax_softmax_3d_dim2": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_softplus_3d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_softplus_4d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_squeeze_single_2d1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_squeeze_single_3d0": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_squeeze_single_3d1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_squeeze_single_3d2": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_squeeze_single_4d0": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_squeeze_single_4d1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_squeeze_single_4d2": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_squeeze_single_4d3": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_squeeze_reduction_sum_4d1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_where_eq_1d256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_where_ge_1d256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_where_gt_1d256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_where_le_1d256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_where_lt_1d256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_where_ne_1d256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_zeros_aligned": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_add_1d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_add_2d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_add_3d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_add_4d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_combined_1d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_combined_2d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_combined_3d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_combined_4d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_div_1d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_div_2d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_div_3d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_div_4d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_mul_1d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_mul_2d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_mul_3d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_mul_4d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_sub_1d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_sub_2d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_sub_3d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_sub_4d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_true_divide_1d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_true_divide_2d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_true_divide_3d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_scalar_cpu_true_divide_4d": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_sdpa_mha_prefill_causal": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_sdpa_mha_prefill": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_sdpa_mha_prefill_mask": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_sum_keepdim0_sum_3d_dim_1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_sum_keepdim0_sum_3d_dim_2": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_sum_keepdim1_sum_3d_dim_1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_sum_keepdim1_sum_3d_dim_2": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_t_2d_contiguous_4096x49280": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_t_2d_contiguous_49280x4096": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_transpose_2d_contiguous_dim_0_1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_transpose_2d_contiguous_dim_0_2": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_transpose_2d_contiguous_dim_0_2_same_dim": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_transpose_2d_contiguous_dim_1_2": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_transpose_3d_contiguous_dim_0_1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_transpose_3d_contiguous_dim_0_2": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_transpose_3d_contiguous_dim_0_2_same_dim": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_transpose_3d_contiguous_dim_1_2": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_transpose_4d_contiguous_dim_0_1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_transpose_4d_contiguous_dim_0_3": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_transpose_4d_contiguous_dim_1_2": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_transpose_4d_contiguous_dim_1_3": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_transpose_4d_contiguous_dim_2_3": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_unsqueeze_single_4d2": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_unsqueeze_broadcast_add_1d0": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_unsqueeze_broadcast_add_2d0": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_unsqueeze_broadcast_add_2d1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_unsqueeze_broadcast_add_3d0": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_unsqueeze_broadcast_add_3d1": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_unsqueeze_broadcast_add_3d2": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_binary_op_add_256_256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_binary_op_add_67x256_67x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_binary_op_add_67x71x256_67x71x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_binary_op_add_7x12x32x64_7x12x32x64": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_binary_op_div_256_256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_binary_op_div_67x256_67x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_binary_op_div_67x71x256_67x71x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_binary_op_div_7x12x32x64_7x12x32x64": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_binary_op_fp32_add_fp32": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_binary_op_fp32_div_fp32": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_binary_op_fp32_mul_fp32": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_binary_op_fp32_sub_fp32": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_binary_op_mul_256_256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_binary_op_mul_67x256_67x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_binary_op_mul_67x71x256_67x71x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_binary_op_mul_7x12x32x64_7x12x32x64": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_binary_op_sub_256_256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_binary_op_sub_67x256_67x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_binary_op_sub_67x71x256_67x71x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_binary_op_sub_7x12x32x64_7x12x32x64": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_unary_op_abs_67x71x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_unary_op_exp_67x71x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_unary_op_neg_67x71x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_unary_op_reciprocal_67x71x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_unary_op_relu_67x71x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_pointwise_unary_op_tanh_67x71x256": TestFailure(
        ("lx_planning_pointwise_subtraction"), is_skip=True
    ),
    "test_rmsnorm_2d": TestFailure(("lx_planning_pointwise_subtraction"), is_skip=True),
    "test_rmsnorm_3d": TestFailure(("lx_planning_pointwise_subtraction"), is_skip=True),
    "test_rmsnorm_4d": TestFailure(("lx_planning_pointwise_subtraction"), is_skip=True),
}


class LxPlanningTwoOpPointwiseSubtractionTest(unittest.TestCase):
    def wrap_pointwise(self, fn):
        @functools.wraps(fn)
        def make_seq_of_ops(*fn_args, **fn_kwargs):
            result = fn(*fn_args, **fn_kwargs)
            return pytree.tree_map(
                lambda x: x - x if isinstance(x, torch.Tensor) else x, result
            )

        return make_seq_of_ops

    def compare_with_cpu(self, fn, *args, **kwargs):
        kwargs["cpu_compile"] = False
        return compare_with_cpu(self.wrap_pointwise(fn), *args, **kwargs)

    def compare(
        self,
        fn,
        *args,
        atol=0.0,
        rtol=0.0,
        cpu_atol=0.1,
        cpu_rtol=0.1,
        needs_device=False,
    ):
        return compare_with_cpu(
            self.wrap_pointwise(fn),
            *args,
            atol=cpu_atol,
            rtol=cpu_rtol,
            needs_device=needs_device,
            cpu_compile=False,
        )


copy_tests(
    make_lx_planning_class(inductor.test_inductor_ops.TestOps),
    LxPlanningTwoOpPointwiseSubtractionTest,
    "lx_planning_pointwise_subtraction",
    POINTWISE_SUBTRACTION_TEST_FAILURES
    if not config.tests_lx_planning_run_skips
    else None,
)


REDUCTION_TEST_FAILURES = {
    "test_add_broadcast_cpu_256_67x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_addmm_1152_10x1152_1152x1152": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_addmm_out_basic": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_addmm_scaled_alpha_0_5": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_alias_operands_cpu_pow_67x71x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_alias_operands_cube_256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_alias_operands_cube_67x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_alias_operands_cube_67x71x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_alias_operands_double_67x71x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_alias_operands_square_67x71x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_alias_operands_triple_256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_alias_operands_triple_67x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_alias_operands_triple_67x71x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_attention_3d_batch_size_1": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_attention_3d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_attention_4d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_bmm_bmm_2x256x1_2x1x128": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_bmm_bmm_2x55x2_2x2x99": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_bmm_bmm_2x99x65_2x65x55": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_bmm_bmm_3x17x256_3x256x128": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_bmm_bmm_3x1x256_3x256x128": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_cat_1d_dim0": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cat_1d_dim0_three_tensors": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_cat_2d_dim0_diff_size": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cat_2d_dim0_three_tensors": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_cat_2d_dim1_diff_size": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cat_3d_dim0": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cat_3d_dim1": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cat_3d_dim1_size1": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cat_3d_dim2": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cat_4d_dim0": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cat_4d_dim1": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cat_4d_dim2": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cat_4d_dim3_fp32": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cat_4d_dim3": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_eq_1d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_eq_2d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_eq_3d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_eq_broadcast": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_ge_1d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_ge_2d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_ge_3d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_ge_broadcast": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_gt_1d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_gt_2d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_gt_3d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_gt_broadcast": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_le_1d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_le_2d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_le_3d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_le_broadcast": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_lt_1d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_lt_2d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_lt_3d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_lt_broadcast": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_ne_1d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_ne_2d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_ne_3d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_cmp_ne_broadcast": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_copy_roundtrip_4d_stick": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_fallback_1d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_fallback_2d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_fallback_3d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_full_value_1": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_full_value_2": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_inplace_op_add_1d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_inplace_op_add_2d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_inplace_op_add_3d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_inplace_op_mul_1d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_inplace_op_mul_2d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_inplace_op_mul_3d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_item_from_computation": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_layernorm_2d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_linear_2d_bias": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_linear_2d_no_bias": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_linear_3d_bias": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_linear_3d_no_bias": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_logical_not_logical_not_1d_fp16": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_logical_not_logical_not_2d_fp16": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_logical_not_logical_not_3d_fp16": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_logical_not_logical_not_4d_fp16": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_logical_not_logical_not_fp16_single_elem": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_matmul_matmul_2x3x55x2_2x3x2x99": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_matmul_matmul_2x3x99x1_2x3x1x55": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_matmul_matmul_2x3x99x65_2x3x65x55": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_matmul_matmul_2x55x2_2x2x99": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_matmul_matmul_2x64x128_128x16384": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_matmul_matmul_2x99x1_1x55": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_matmul_matmul_2x99x1_2x1x55": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_matmul_matmul_2x99x65_2x65x55": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_matmul_matmul_3x17x256_3x256x128": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_matmul_matmul_3x18x128x256_3x18x256x128": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_matmul_matmul_3x1x256_3x256x128": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_matmul_matmul_512x256_256x128": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_matmul_matmul_55x2_2x99": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_matmul_matmul_99x1_1x55": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_matmul_matmul_99x65_65x55": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_keepdim0_sum_2d_dim_0": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_keepdim0_sum_2d_dim_1": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_keepdim0_sum_3d_dim_1": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_keepdim0_sum_3d_dim_2": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_keepdim0_sum_4d_dim_0": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_keepdim0_sum_4d_dim_1": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_keepdim0_sum_4d_dim_2": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_keepdim0_sum_4d_dim_3": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_keepdim1_sum_2d_dim_0": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_keepdim1_sum_2d_dim_1": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_keepdim1_sum_3d_dim_0": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_keepdim1_sum_3d_dim_1": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_keepdim1_sum_3d_dim_2": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_keepdim1_sum_4d_dim_0": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_keepdim1_sum_4d_dim_1": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_keepdim1_sum_4d_dim_2": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_keepdim1_sum_4d_dim_3": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_sub_broadcast_2d_dim_0": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_sub_broadcast_2d_dim_1": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_sub_broadcast_4d_dim_0": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_sub_broadcast_4d_dim_1": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_sub_broadcast_4d_dim_2": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_max_sub_broadcast_4d_dim_3": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_mm_mm_55x2_2x99": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_mm_mm_67x255_255x128": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_mm_mm_67x256_256x128": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_mm_mm_67x67_67x67": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_pad_2d_both_dims": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_pad_2d_dim0_left": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_pad_2d_dim0_left_only": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_pad_2d_last_dim_left_and_right_stick_aligned": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pad_2d_last_dim_left_stick_aligned": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pad_2d_last_dim_left_two_sticks": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pad_2d_last_dim_right": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_pad_3d_dim0_left": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_pad_3d_dim1_left": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_pad_3d_dim1_right": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_pad_3d_last_dim_right": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_pad_4d_dim0_left": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_permute_3d_0_2_1": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_permute_4d_0_3_1_2": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_permute_4d_0_m2_m1_1": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_permute_5d_0_2_3_4_1": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_pointwise_binary_op_add_256_256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_binary_op_add_67x256_67x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_binary_op_add_67x71x256_67x71x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_binary_op_add_7x12x32x64_7x12x32x64": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_binary_op_div_256_256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_binary_op_div_67x256_67x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_binary_op_div_67x71x256_67x71x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_binary_op_div_7x12x32x64_7x12x32x64": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_binary_op_fp32_add_fp32": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_binary_op_fp32_div_fp32": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_binary_op_fp32_mul_fp32": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_binary_op_fp32_sub_fp32": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_binary_op_mul_256_256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_binary_op_mul_67x256_67x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_binary_op_mul_67x71x256_67x71x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_binary_op_mul_7x12x32x64_7x12x32x64": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_binary_op_sub_256_256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_binary_op_sub_67x256_67x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_binary_op_sub_67x71x256_67x71x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_binary_op_sub_7x12x32x64_7x12x32x64": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_range_op_clamp_fp16": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_unary_op_abs_67x71x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_unary_op_exp_67x71x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_unary_op_neg_67x71x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_unary_op_reciprocal_67x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_unary_op_reciprocal_67x71x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_unary_op_relu_67x71x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_pointwise_unary_op_tanh_67x71x256": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_rmsnorm_2d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_rmsnorm_3d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_rmsnorm_4d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_scalar_cpu_add_1d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_scalar_cpu_add_2d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_scalar_cpu_add_3d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_scalar_cpu_add_4d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_scalar_cpu_combined_1d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_scalar_cpu_combined_2d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_scalar_cpu_combined_3d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_scalar_cpu_combined_4d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_scalar_cpu_div_1d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_scalar_cpu_div_2d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_scalar_cpu_div_3d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_scalar_cpu_div_4d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_scalar_cpu_mul_1d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_scalar_cpu_mul_2d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_scalar_cpu_mul_3d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_scalar_cpu_mul_4d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_scalar_cpu_sub_1d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_scalar_cpu_sub_2d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_scalar_cpu_sub_3d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_scalar_cpu_sub_4d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_scalar_cpu_true_divide_1d": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_scalar_cpu_true_divide_2d": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_scalar_cpu_true_divide_3d": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_scalar_cpu_true_divide_4d": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_sdpa_mha_prefill_causal": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_sdpa_mha_prefill": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_sdpa_mha_prefill_mask": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_softmax_softmax_2d_dim0": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_softmax_softmax_2d_dim1": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_softmax_softmax_3d_dim0": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_softmax_softmax_3d_dim1": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_softmax_softmax_3d_dim2": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_softplus_3d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_softplus_4d": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_squeeze_reduction_sum_3d0": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_squeeze_reduction_sum_4d0": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_squeeze_single_3d0": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_squeeze_single_4d0": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_sum_keepdim0_sum_3d_dim_1": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_sum_keepdim0_sum_3d_dim_2": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_sum_keepdim1_sum_3d_dim_1": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_sum_keepdim1_sum_3d_dim_2": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_t_2d_contiguous_1088x320": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_t_2d_contiguous_320x320": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_t_2d_contiguous_4096x49280": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_t_2d_contiguous_49280x4096": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_transpose_2d_contiguous_dim_0_1": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_transpose_2d_contiguous_dim_0_2": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_transpose_2d_contiguous_dim_0_2_same_dim": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_transpose_2d_contiguous_dim_1_2": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_transpose_2d_dim_1_2": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_transpose_3d_contiguous_dim_0_1": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_transpose_3d_contiguous_dim_0_2": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_transpose_3d_contiguous_dim_0_2_same_dim": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_transpose_3d_contiguous_dim_1_2": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_transpose_3d_dim_1_2": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_transpose_4d_contiguous_dim_0_1": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_transpose_4d_contiguous_dim_0_3": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_transpose_4d_contiguous_dim_1_2": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_transpose_4d_contiguous_dim_1_3": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_transpose_4d_contiguous_dim_2_3": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_transpose_4d_dim_1_2": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_transpose_4d_dim_1_3": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_transpose_4d_dim_2_3": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_unsqueeze_broadcast_add_1d0": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_unsqueeze_broadcast_add_2d0": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_unsqueeze_broadcast_add_2d1": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_unsqueeze_broadcast_add_3d0": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_unsqueeze_broadcast_add_3d1": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_unsqueeze_broadcast_add_3d2": TestFailure(
        ("lx_planning_reduction"), is_skip=True
    ),
    "test_where_eq_1d256": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_where_ge_1d256": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_where_gt_1d256": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_where_le_1d256": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_where_lt_1d256": TestFailure(("lx_planning_reduction"), is_skip=True),
    "test_where_ne_1d256": TestFailure(("lx_planning_reduction"), is_skip=True),
}


class LxPlanningTwoOpReductionTest(unittest.TestCase):
    def wrap_reduction(self, fn):
        @functools.wraps(fn)
        def make_seq_of_ops(*fn_args, **fn_kwargs):
            result = fn(*fn_args, **fn_kwargs)
            return pytree.tree_map(
                lambda x: torch.sum(x, dim=0)
                if isinstance(x, torch.Tensor) and x.dtype == torch.float16
                else x,
                result,
            )

        return make_seq_of_ops

    def compare_with_cpu(self, fn, *args, **kwargs):
        kwargs["cpu_compile"] = False
        return compare_with_cpu(self.wrap_reduction(fn), *args, **kwargs)

    def compare(
        self,
        fn,
        *args,
        atol=0.0,
        rtol=0.0,
        cpu_atol=0.1,
        cpu_rtol=0.1,
        needs_device=False,
    ):
        return compare_with_cpu(
            self.wrap_reduction(fn),
            *args,
            atol=cpu_atol,
            rtol=cpu_rtol,
            needs_device=needs_device,
            cpu_compile=False,
        )


copy_tests(
    make_lx_planning_class(inductor.test_inductor_ops.TestOps),
    LxPlanningTwoOpReductionTest,
    "lx_planning_reduction",
    REDUCTION_TEST_FAILURES if not config.tests_lx_planning_run_skips else None,
)

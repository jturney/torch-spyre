/*
 * Copyright 2025 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "spyre_views.h"

#include <ATen/EmptyTensor.h>
#include <ATen/InferSize.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/as_strided_cpu_dispatch.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>
#include <torch/csrc/inductor/inductor_ops.h>
#include <torch/library.h>
#include <util/sen_data_convert.h>

#include <vector>

#include "spyre_tensor_impl.h"

namespace spyre {

namespace {
// A reshape/view that changes the innermost (stick) dimension is stick-layout
// incompatible on Spyre: the new element->stick mapping needs a cross-stick
// gather the device addressing model cannot express (the same gap that makes
// flatten/reshape fail when consumed by a compute kernel). Until an on-device
// restickify gather exists, materialize such a view through a CPU round-trip --
// the same stopgap the resize_ reallocate path uses (spyre_mem.cpp Case 3) --
// so downstream kernels receive a stick-aligned tensor.
//
// This gives up view aliasing for these specific views, which is unavoidable: a
// stick-incompatible reshape can never be a correct device alias (the data is
// not physically tiled that way), so materializing is strictly better than
// returning an unreadable alias.
constexpr int64_t kStickBytes = 128;

bool reshape_changes_stick_dim(const at::Tensor& self,
                               c10::IntArrayRef new_size) {
  if (self.dim() == 0 || new_size.empty()) {
    return false;
  }
  const int64_t old_inner = self.size(self.dim() - 1);
  const int64_t new_inner = new_size.back();
  if (new_inner == old_inner) {
    return false;
  }
  // A size-1 innermost is a degenerate squeeze/unsqueeze (e.g. (67,256,1) ->
  // (1,67,256)), not a real stick change -- it is representable and must not be
  // materialized (doing so corrupts the data). Skip it.
  if (old_inner <= 1 || new_inner <= 1) {
    return false;
  }
  // Otherwise representable iff BOTH innermosts are stick-aligned (multiples of
  // elems_per_stick): the reshape stays on stick boundaries, so no cross-stick
  // gather is needed (e.g. a 384 -> 64 split, or a 64 -> 128 merge of full
  // sticks). If either side is a partial stick, the reindex crosses a padded
  // stick boundary and needs a gather -> materialize.
  const int64_t elems_per_stick = kStickBytes / self.element_size();
  if (elems_per_stick <= 0) {
    return false;
  }
  return old_inner % elems_per_stick != 0 || new_inner % elems_per_stick != 0;
}

at::Tensor materialize_reshape_via_cpu(const at::Tensor& self,
                                       c10::IntArrayRef new_size) {
  TORCH_WARN_ONCE(
      "Spyre: stick-incompatible reshape/view (innermost ",
      self.size(self.dim() - 1), " -> ", new_size.back(),
      ") materialized via a CPU round-trip; correctness fallback with a "
      "device<->host copy cost until an on-device restickify gather exists.");
  return self.cpu().reshape(new_size).to(self.device());
}

at::Tensor materialize_unfold_via_cpu(const at::Tensor& self, int64_t dimension,
                                      int64_t size, int64_t step) {
  TORCH_WARN_ONCE(
      "Spyre: unfold (size ", size, ", step ", step,
      ") materialized via a CPU round-trip; correctness fallback with a "
      "device<->host copy cost. Gives up unfold view aliasing, which is "
      "unavoidable: unfold's two output dims index the same storage with "
      "overlapping strides, so an on-device read of the alias mis-addresses.");
  return self.cpu()
      .unfold(dimension, size, step)
      .contiguous()
      .to(self.device());
}
}  // namespace

//
// templated for ArrayRef<int64_t> and SmallVector<int64_t> use cases
//
template <typename Vec>
static at::Tensor spyre_alias_with_sizes_and_strides(const at::Tensor& self,
                                                     const Vec& sizes,
                                                     const Vec& strides) {
  // caller should make sure that sizes and strides are valid for self
  // (storage is sufficient, strides are non-negative, strides and sizes array
  // size is the same)
  auto orig_impl = static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl());
  SpyreTensorLayout stl = orig_impl->spyre_layout;
  at::Tensor self_;
  self_ = at::detail::make_tensor<SpyreTensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
      self.dtype());
  auto spyre_tensor_impl_ =
      static_cast<SpyreTensorImpl*>(self_.unsafeGetTensorImpl());
  spyre_tensor_impl_->set_storage_offset(self.storage_offset());
  spyre_tensor_impl_->set_sizes_and_strides(sizes, strides);
  spyre_tensor_impl_->spyre_layout = stl;
  spyre_tensor_impl_->dma_sizes = orig_impl->dma_sizes;
  spyre_tensor_impl_->dma_strides = orig_impl->dma_strides;
  return self_;
}

// specialization for symbolic shapes and strides.
// SymIntArrayRef/ArrayRef<c10::SymInt> and
// SmallVector<c10::SymInt>/SymDimVector
template <template <typename...> typename Container>
static at::Tensor spyre_alias_with_sizes_and_strides(
    const at::Tensor& self, const Container<c10::SymInt>& sizes,
    const Container<c10::SymInt>& strides) {
  // caller should make sure that sizes and strides are valid for self
  // (storage is sufficient, strides are non-negative, strides and sizes array
  // size is the same)
  auto orig_impl = static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl());
  SpyreTensorLayout stl = orig_impl->spyre_layout;
  at::Tensor self_;
  self_ = at::detail::make_tensor<SpyreTensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
      self.dtype());
  auto spyre_tensor_impl_ =
      static_cast<SpyreTensorImpl*>(self_.unsafeGetTensorImpl());
  spyre_tensor_impl_->set_sizes_and_strides(sizes, strides,
                                            self.sym_storage_offset());
  spyre_tensor_impl_->spyre_layout = stl;
  spyre_tensor_impl_->dma_sizes = orig_impl->dma_sizes;
  spyre_tensor_impl_->dma_strides = orig_impl->dma_strides;
  return self_;
}

static inline at::Tensor spyre_view_impl(const at::Tensor& self,
                                         c10::IntArrayRef size) {
  c10::DimVector inferred_size = at::infer_size_dv(size, self.numel());
  if (reshape_changes_stick_dim(self, inferred_size)) {
    return materialize_reshape_via_cpu(self, inferred_size);
  }
  auto stride =
      at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
  TORCH_CHECK(
      stride.has_value(),
      "view size is "
      "not compatible with input tensor's size and stride (at least one "
      "dimension"
      " spans across two contiguous subspaces). Use .reshape(...) instead.");
  return spyre_alias_with_sizes_and_strides(self, inferred_size, *stride);
}

at::Tensor spyre_view(const at::Tensor& self, c10::IntArrayRef size) {
  return spyre_view_impl(self, size);
}

at::Tensor spyre__unsafe_view(const at::Tensor& self, c10::IntArrayRef size) {
  return spyre_view_impl(self, size);
}

at::Tensor spyre_reshape_alias(const at::Tensor& self, c10::IntArrayRef sizes,
                               c10::IntArrayRef strides) {
  if (reshape_changes_stick_dim(self, sizes)) {
    return materialize_reshape_via_cpu(self, sizes);
  }
  return spyre_alias_with_sizes_and_strides(self, sizes, strides);
}

at::Tensor spyre_as_strided(const at::Tensor& self, c10::IntArrayRef size,
                            c10::IntArrayRef stride,
                            std::optional<int64_t> storage_offset_) {
  SpyreTensorLayout stl =
      (static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl()))->spyre_layout;
  return as_strided_with_layout(self, size, stride, storage_offset_, stl);
}

at::Tensor as_strided_with_layout(const at::Tensor& self, c10::IntArrayRef size,
                                  c10::IntArrayRef stride,
                                  std::optional<int64_t> storage_offset_,
                                  SpyreTensorLayout device_layout) {
  auto orig_impl = static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl());
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  auto result = at::detail::make_tensor<SpyreTensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
      self.dtype());
  at::native::setStrided(result, size, stride, storage_offset);
  auto spyre_impl = static_cast<SpyreTensorImpl*>(result.unsafeGetTensorImpl());
  spyre_impl->spyre_layout = device_layout;
  if (device_layout == orig_impl->spyre_layout) {
    spyre_impl->dma_sizes = orig_impl->dma_sizes;
    spyre_impl->dma_strides = orig_impl->dma_strides;
  } else {
    spyre_impl->dma_sizes = size.vec();
    spyre_impl->dma_strides = stride.vec();
  }

  return result;
}

// Similar to as_strided with the following differences
// - offset is added to the existing offset (rather than replacing it)
// - view tracking is disabled similar to unsafe_view
at::Tensor reinterpret_tensor(const at::Tensor& self, c10::IntArrayRef size,
                              c10::IntArrayRef stride,
                              int64_t offset_increment) {
  // For in-tree devices (e.g. CPU tensors carried by FallbackKernels), there is
  // no SpyreTensorImpl to reinterpret, so defer to the stock Inductor helper.
  if (self.device().type() != c10::DeviceType::PrivateUse1) {
    return torch::inductor::_reinterpret_tensor(self, size, stride,
                                                offset_increment);
  }
  auto orig_impl = static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl());
  SpyreTensorLayout stl = orig_impl->spyre_layout;
  return reinterpret_tensor_with_layout(self, size, stride, offset_increment,
                                        stl);
}

at::Tensor reinterpret_tensor_with_layout(const at::Tensor& self,
                                          c10::IntArrayRef size,
                                          c10::IntArrayRef stride,
                                          int64_t offset_increment,
                                          SpyreTensorLayout stl) {
  // Purely defensive: If a non-Spyre tensor ever arrives, fall back to the
  // stock Inductor helper.
  if (self.device().type() != c10::DeviceType::PrivateUse1) {
    return torch::inductor::_reinterpret_tensor(self, size, stride,
                                                offset_increment);
  }
  auto orig_impl = static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl());
  SpyreTensorLayout orig_stl = orig_impl->spyre_layout;
  at::Tensor self_ = at::detail::make_tensor<SpyreTensorImpl>(
      c10::Storage(self.storage()), self.key_set(), self.dtype());
  auto* spyre_tensor_impl_ =
      static_cast<SpyreTensorImpl*>(self_.unsafeGetTensorImpl());
  spyre_tensor_impl_->set_storage_offset(self.storage_offset() +
                                         offset_increment);
  spyre_tensor_impl_->set_sizes_and_strides(size, stride);
  spyre_tensor_impl_->spyre_layout = stl;
  if (stl == orig_stl) {
    spyre_tensor_impl_->dma_sizes = orig_impl->dma_sizes;
    spyre_tensor_impl_->dma_strides = orig_impl->dma_strides;
  } else {
    spyre_tensor_impl_->dma_sizes = size.vec();
    spyre_tensor_impl_->dma_strides = stride.vec();
  }
  return self_;
}

at::Tensor spyre_alias(const at::Tensor& self) {
  return spyre_alias_with_sizes_and_strides(self, self.sym_sizes(),
                                            self.sym_strides());
}

at::Tensor spyre_unfold(const at::Tensor& self, int64_t dimension, int64_t size,
                        int64_t step) {
  // Normalize negative dimension
  auto ndim = self.dim();
  dimension = c10::maybe_wrap_dim(dimension, ndim);

  // Validate parameters
  auto dim_size = self.size(dimension);
  TORCH_CHECK(size > 0, "unfold: size must be positive, got ", size);
  TORCH_CHECK(step > 0, "unfold: step must be positive, got ", step);
  TORCH_CHECK(size <= dim_size, "unfold: maximum size for tensor at dimension ",
              dimension, " is ", dim_size, " but size is ", size);

  // An unfold with window size > 1 produces two output dims (num_slices @
  // stride*step, window @ stride) that index the same parent storage; an
  // on-device read of that overlapping/strided view mis-addresses (wrong values
  // or an unsupported multi-variable stick expression). Materialize to a
  // compact buffer instead of returning an aliasing view. A size-1 window is a
  // no-op view that addresses fine and stays a view. Mirrors the
  // reshape_changes_stick_dim guard in spyre_view_impl; the compiled path has
  // the equivalent reroute_overlapping_unfold pass.
  if (size > 1) {
    return materialize_unfold_via_cpu(self, dimension, size, step);
  }

  // Compute new sizes
  std::vector<int64_t> new_sizes(self.sizes().begin(), self.sizes().end());
  int64_t num_slices = (dim_size - size) / step + 1;
  new_sizes[dimension] = num_slices;
  new_sizes.push_back(size);

  // Compute new strides
  std::vector<int64_t> new_strides(self.strides().begin(),
                                   self.strides().end());
  int64_t original_stride = self.stride(dimension);
  new_strides[dimension] = original_stride * step;
  new_strides.push_back(original_stride);

  auto orig_impl = static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl());
  at::Tensor result = at::detail::make_tensor<SpyreTensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
      self.dtype());
  at::native::setStrided(result, c10::IntArrayRef(new_sizes),
                         c10::IntArrayRef(new_strides), self.storage_offset());

  auto* result_impl =
      static_cast<SpyreTensorImpl*>(result.unsafeGetTensorImpl());
  result_impl->spyre_layout = orig_impl->spyre_layout;
  result_impl->dma_sizes = orig_impl->dma_sizes;
  result_impl->dma_strides = orig_impl->dma_strides;

  return result;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("view", TORCH_FN(spyre_view));
  m.impl("_unsafe_view", TORCH_FN(spyre__unsafe_view));
  m.impl("_reshape_alias", TORCH_FN(spyre_reshape_alias));
  m.impl("alias", TORCH_FN(spyre_alias));
  m.impl("as_strided", TORCH_FN(spyre_as_strided));
  m.impl("unfold", TORCH_FN(spyre_unfold));
}

}  // namespace spyre

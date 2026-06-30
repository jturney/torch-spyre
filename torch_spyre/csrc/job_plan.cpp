/*
 * Copyright 2026 The Torch-Spyre Authors.
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

#include "job_plan.h"

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "spyre_allocator.h"
#include "spyre_stream.h"
#include "spyre_tensor_impl.h"
#include "util/processSpyreCodeArtifacts.h"

namespace spyre {

namespace {
// A view tensor shares its parent storage's composite_addr (the parent base);
// the per-view element offset lives only in tensor.storage_offset(). The kernel
// launch must fold that offset into the device address, otherwise every view
// reads the parent base (e.g. all unbind slices read slice 0).
//
// storage_offset is a flat HOST (logical) offset; the device storage is tiled,
// so it does NOT map linearly to device bytes -- the tiling reorders sticks.
// The translation uses the view's physical layout
// (SpyreTensorImpl::spyre_layout, which is the layout the storage was written
// with, i.e. the parent's tiling): decompose the host offset into device
// coordinates via stride_map (each stride_map[d] is the host stride of device
// dim d) and accumulate device contiguous strides. A stick is 128 bytes;
// whole-stick offsets (non-stick selects) land stick-aligned, sub-stick offsets
// (a select/unbind along the stick dim) do not and are rejected (Phase 2 /
// restickify).
constexpr int64_t kStickBytes = 128;

int64_t view_device_byte_offset(const at::Tensor& tensor) {
  const int64_t offset_elems = tensor.storage_offset();
  if (offset_elems == 0) {
    return 0;
  }
  const auto* impl =
      static_cast<const SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  const std::vector<int64_t>& dsize = impl->spyre_layout.device_size;
  const std::vector<int64_t>& smap = impl->spyre_layout.stride_map;
  const int rank = static_cast<int>(dsize.size());
  TORCH_CHECK(static_cast<int>(smap.size()) == rank,
              "Spyre: device_size/stride_map rank mismatch");

  // Device contiguous strides (in elements): the device storage is laid out
  // contiguously in device_size order.
  std::vector<int64_t> dstride(rank, 1);
  for (int d = rank - 2; d >= 0; --d) {
    dstride[d] = dstride[d + 1] * dsize[d + 1];
  }

  int64_t elem_off = 0;
  for (int d = 0; d < rank; ++d) {
    if (dsize[d] <= 1 || smap[d] == 0) {
      continue;
    }
    const int64_t coord = (offset_elems / smap[d]) % dsize[d];
    elem_off += coord * dstride[d];
  }
  const int64_t byte_off = elem_off * tensor.element_size();
  TORCH_CHECK(
      byte_off % kStickBytes == 0,
      "Spyre: kernel input has a sub-stick view offset (storage_offset ",
      offset_elems, " elems -> ", byte_off,
      " device bytes, stick = ", kStickBytes,
      "); only stick-aligned view offsets are supported (a select/"
      "unbind along the stick dimension is not yet supported).");
  return byte_off;
}

const flex::CompositeAddress& base_composite_address(const at::Tensor& tensor) {
  return static_cast<SharedOwnerCtx*>(tensor.storage().data_ptr().get_context())
      ->composite_addr;
}

// Build a single-chunk composite address shifted by byte_off (>0). The
// no-offset path passes the base by reference instead (CompositeAddress has a
// deleted copy ctor, and the base may be multi-chunk).
flex::CompositeAddress shifted_composite_address(
    const flex::CompositeAddress& base, int64_t byte_off) {
  TORCH_CHECK(base.chunks().size() == 1,
              "Spyre: offset view on an interleaved composite address is not "
              "supported");
  const auto& base_chunk = base.chunks()[0];
  flex::LogicalAddress shifted_addr(base_chunk.addr.region_id,
                                    base_chunk.addr.offset + byte_off);
  flex::Chunk shifted_chunk(shifted_addr, base.total_size() - byte_off,
                            base_chunk.domain_id);
  return flex::CompositeAddress(shifted_chunk);
}
}  // namespace

void JobPlanStepH2D::construct(LaunchContext&,
                               const SpyreStream& stream) const {
  auto* params =
      flex::createDmaParams(host_address_, device_address_.total_size(),
                            /*to_device=*/true, &device_address_);
  params->pipeline_barrier = pipeline_barrier_;
  stream.launchH2D(params);
  flex::destroyDmaParams(params);
}

void JobPlanStepH2D::write(std::ostream& os) const {
  os << "  H2D (Host-to-Device)\n";
  os << "    Host address: " << host_address_ << "\n";
  os << "    Device address: " << device_address_ << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

void JobPlanStepD2H::construct(LaunchContext&,
                               const SpyreStream& stream) const {
  auto* params =
      flex::createDmaParams(host_address_, device_address_.total_size(),
                            /*to_device=*/false, &device_address_);
  params->pipeline_barrier = pipeline_barrier_;
  stream.launchD2H(params);
  flex::destroyDmaParams(params);
}

void JobPlanStepD2H::write(std::ostream& os) const {
  os << "  D2H (Device-to-Host)\n";
  os << "    Device address: " << device_address_ << "\n";
  os << "    Host address: " << host_address_ << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

void JobPlanStepCompute::construct(LaunchContext& ctx,
                                   const SpyreStream& stream) const {
  std::vector<const flex::CompositeAddress*> tensor_allocs;
  // Backs the offset-shifted addresses below; reserved so push_back never
  // reallocates and invalidates the pointers held in tensor_allocs.
  std::vector<flex::CompositeAddress> owned_addrs;
  if (bind_io_addresses_) {
    owned_addrs.reserve(ctx.inputs_outputs.size());
    for (auto& tensor : ctx.inputs_outputs) {
      const flex::CompositeAddress& base = base_composite_address(tensor);
      const int64_t byte_off = view_device_byte_offset(tensor);
      if (byte_off == 0) {
        tensor_allocs.push_back(&base);
      } else {
        owned_addrs.push_back(shifted_composite_address(base, byte_off));
        tensor_allocs.push_back(&owned_addrs.back());
      }
    }
  }
  auto* params = flex::createComputeParams(
      &program_address_, std::move(tensor_allocs), name_, bootstrap_offset_);
  params->pipeline_barrier = pipeline_barrier_;
  stream.launchCompute(params);
  flex::destroyComputeParams(params);
}

void JobPlanStepCompute::write(std::ostream& os) const {
  os << "  Device Compute\n";
  os << "    Name: " << (name_.empty() ? "(unnamed)" : name_) << "\n";
  os << "    Program address: " << program_address_ << "\n";
  os << "    Bind I/O addresses: " << (bind_io_addresses_ ? "yes" : "no")
     << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

// TODO(jni): move to flex
// convert CompositeAddress to dmva
static int64_t composite_address_to_dmva(
    const flex::CompositeAddress& composite_address) {
  size_t num_chunks = composite_address.chunks().size();
  TORCH_CHECK(num_chunks == 1, "Interleaved not supported yet");

  const auto& addr = composite_address.chunks()[0].addr;
  auto& allocator = SpyreAllocator::instance();
  auto seg_id = allocator.segmentForRegion(addr.region_id);
  auto address = flex::SegmentByteOffset_todmva(seg_id, addr.offset);
  return address;
}

void JobPlanStepHostCompute::construct(LaunchContext& ctx,
                                       const SpyreStream& stream) const {
  // Helper lambda to build HostCallbackParams and launch on the stream
  auto launch_host_callback = [this, &stream](auto&& callback) {
    auto* params = flex::createHostCallbackParams(
        std::forward<decltype(callback)>(callback), nullptr, pipeline_barrier_);
    stream.launchHostCallback(params);
    flex::destroyHostCallbackParams(params);
  };

  // Case 1: input_buffer_ is provided
  if (input_buffer_ != nullptr) {
    launch_host_callback([this](void*) {
      deeptools::processComputeOnHostCommand(*hcm_, output_buffer_,
                                             input_buffer_);
    });
    return;
  }

  // Case 2: fake symbols (ishape_ is {0})
  // Further discussion is required on "ishape". For now, it's vector<int64_t>,
  // and it's {0}, it's for fake symbols
  if (ishape_.size() == 1 && ishape_[0] == 0) {
    launch_host_callback([this](void*) {
      deeptools::processComputeOnHostCommand(*hcm_, output_buffer_, nullptr);
    });
    return;
  }

  // Case 3: extract addresses from context tensors
  std::vector<int64_t> addresses(ctx.inputs_outputs.size());
  int addr_idx = 0;
  for (auto& tensor : ctx.inputs_outputs) {
    const flex::CompositeAddress& base = base_composite_address(tensor);
    const int64_t byte_off = view_device_byte_offset(tensor);
    int64_t addr = byte_off == 0
                       ? composite_address_to_dmva(base)
                       : composite_address_to_dmva(
                             shifted_composite_address(base, byte_off));
    addresses[addr_idx++] = addr;
  }

  launch_host_callback([this, addresses](void*) {
    deeptools::processComputeOnHostCommand(*hcm_, output_buffer_, &addresses);
  });
}

void JobPlanStepHostCompute::write(std::ostream& os) const {
  os << "  Host Compute\n";
  os << "    Output buffer: " << output_buffer_ << "\n";
  os << "    HCM metadata: " << (hcm_ ? "present" : "null") << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

std::ostream& operator<<(std::ostream& os, const JobPlan& plan) {
  os << "============ JobPlan =============\n";
  os << "Total steps: " << plan.steps.size() << "\n";

  // Job allocation
  size_t addr_idx = 0;
  for (const auto& addr : plan.job_allocation) {
    if (addr_idx == 0) {
      os << "Job allocation: " << addr << "\n";
    } else {
      os << "Program " << addr_idx - 1 << ": " << addr << "\n";
    }
    ++addr_idx;
  }

  // Expected input shapes
  if (!plan.expected_input_shapes.empty()) {
    os << "Expected input shapes (" << plan.expected_input_shapes.size()
       << " tensors):\n";
    for (size_t i = 0; i < plan.expected_input_shapes.size(); ++i) {
      os << "  Input " << i << ": [";
      for (size_t j = 0; j < plan.expected_input_shapes[i].size(); ++j) {
        if (j > 0) os << ", ";
        os << plan.expected_input_shapes[i][j];
      }
      os << "]\n";
    }
  }

  // Pinned buffers
  os << "Pinned buffers: " << plan.pinned_buffers.size() << "\n";
  for (size_t i = 0; i < plan.pinned_buffers.size(); ++i) {
    const auto& buf = plan.pinned_buffers[i];
    os << "  Buffer " << i << ": ptr=" << buf.data() << ", size=" << buf.size()
       << " bytes\n";
  }

  // Detailed step information
  os << "\nDetailed Steps:\n";
  for (size_t i = 0; i < plan.steps.size(); ++i) {
    os << "Step " << i << ": ";
    os << *plan.steps[i];
  }

  os << "==================================\n";
  return os;
}

}  // namespace spyre

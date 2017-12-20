#include "comm.h"

#include "torch/csrc/utils/tensor_flatten.h"
#include "torch/csrc/utils/auto_gpu.h"
#include "torch/csrc/cuda/device_set.h"
#include "torch/csrc/cuda/nccl.h"

#include <ATen/ATen.h>

namespace torch { namespace cuda {

using namespace at;

// Some operations can be performed more efficiently if we're handling tensors
// of a single type only. Adding this logic directly in the loop makes it a bit
// ugly, so here's a helper for it.
struct unique_type_checker {
  void show(const at::Type& t) {
    if (!unique) return;
    if (!type) type = &t;
    unique = (type == &t);
  }

  const at::Type *type = nullptr;
  bool unique = true;
};

std::vector<Tensor> broadcast(const Tensor& tensor, const IntList& devices) {
  if (tensor.get_device() != devices[0])
    throw std::runtime_error("device of broadcasted tensor must appear as the "
                             "first on devices list");
  auto & type = tensor.type();
  std::vector<Tensor> tensors;
  tensors.reserve(devices.size());
  tensors.push_back(tensor);
  if (nccl::is_available(tensors)) {
    for (std::size_t i = 1, num_devices = devices.size(); i < num_devices; ++i) {
      AutoGPU _gpu_guard(devices[i]);
      tensors.push_back(type.tensor(tensor.sizes()));
    }
    nccl::broadcast(tensors);
  } else {
    auto & gpu_type = type.toBackend(at::kCUDA);
    for (std::size_t i = 1, num_devices = devices.size(); i < num_devices; ++i) {
      AutoGPU _gpu_guard(devices[i]);
      tensors.push_back(gpu_type.copy(tensor, true));
    }
  }
  return tensors;
}

tensor_list2d broadcast_coalesced(const TensorList& tensors, const IntList& devices, std::size_t buffer_size) {
  if (!std::all_of(tensors.begin(), tensors.end(),
                   [&](const at::Tensor& t) { return t.get_device() == devices[0]; })) {
    throw std::runtime_error("all tensors must be on devices[0]");
  }

  tensor_list2d outputs(devices.size());
  outputs[0] = tensors;
  for (auto & o : outputs)
    o.reserve(tensors.size());

  unique_type_checker type_checker;
  for (auto & chunk : utils::take_tensors(tensors, buffer_size)) {
    auto & type = chunk.type();
    type_checker.show(type);
    if (chunk.type().is_sparse()) {
      // TODO: sparse tensors
      //flat_indices, flat_values = _flatten_sparse_tensors(chunk)
      //result_indices = broadcast(flat_indices, devices)
      //result_values = broadcast(flat_values, devices)
      //unflat_results = tuple(_unflatten_sparse_tensors(iv, chunk) for iv in zip(result_indices, result_values))
      throw std::runtime_error("TODO: sparse broadcast");
    } else {
      std::vector<Tensor> results = broadcast(utils::flatten_dense_tensors(chunk.tensors),
                                              devices);
      for (std::size_t i = 1, num_devices = devices.size(); i < num_devices; ++i) {
        auto & device_outputs = outputs[i];
        for (auto t : utils::unflatten_dense_tensors(results[i], chunk.tensors)) {
          device_outputs.push_back(std::move(t));
        }
      }
    }
  }

  // If we only saw a single tensor type, then we can skip expensive reordering
  if (!type_checker.unique) {
    for (auto & o : outputs)
      utils::reorder_tensors_like(o, tensors);
  }
  return outputs;
}

}}

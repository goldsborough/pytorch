#include "torch/csrc/utils/tensor_flatten.h"

#include <unordered_map>

namespace torch { namespace utils {

using namespace at;

std::vector<TensorGroup> take_tensors(const TensorList& tensors, std::size_t size_limit) {
  std::vector<TensorGroup> results;
  results.reserve(tensors.size()); // an overapproximation, but at least we won't have to copy stuff around
  std::unordered_map<at::Type*, TensorGroup> groups_;
  for (const auto & tensor : tensors) {
    auto & type = tensor.type();
    std::size_t tensor_size;
    if (type.is_sparse()) {
      // TODO: sparse tensors
      //indices = tensor._indices()
      //values = tensor._values()
      //size = indices.numel() * indices.element_size() + values.numel() * values.element_size()
      throw std::runtime_error("TODO: sprase tensors in take_tensors");
    } else {
      tensor_size = tensor.numel() * type.elementSizeInBytes();
    }
    auto & type_group = groups_[&type];
    type_group.tensors.push_back(tensor);
    type_group.size += tensor_size;
    if (type_group.size + tensor_size >= size_limit) {
      results.emplace_back();
      std::swap(results.back(), type_group);
    }
  }
  // End case. Look for any remaining groups and return them.
  for (auto & entry : groups_) {
    auto & group = entry.second;
    if (group.size > 0) {
      results.emplace_back(std::move(group));
    }
  }
  return results;
}

void reorder_tensors_like(std::vector<Tensor>& tensors, const TensorList& order) {
  TORCH_ASSERT(tensors.size() == order.size());
  std::unordered_map<at::Type*, std::vector<std::size_t>> type_indices;
  for (std::size_t i = 0, num_tensors = tensors.size(); i < num_tensors; ++i)
    type_indices[&tensors[i].type()].push_back(i);

  std::unordered_map<at::Type*, std::size_t> type_used;
  std::vector<Tensor> ordered_tensors;
  ordered_tensors.reserve(tensors.size());
  for (auto & tmpl_tensor : order) {
    auto * type = &tmpl_tensor.type();
    auto & indices = type_indices[type];
    auto & used = type_used[type];
    ordered_tensors.push_back(tensors[indices[used++]]);
  }
  std::swap(tensors, ordered_tensors);
}

}}

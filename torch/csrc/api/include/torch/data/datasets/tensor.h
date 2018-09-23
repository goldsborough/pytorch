#pragma once

#include <torch/csrc/utils/functional.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/tensor.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace data {
namespace datasets {

// Dataset specialization for Example<>; collates a batch of tensors into a
// single tensor.
template <typename S>
class Dataset<S, Example<>, Example<>> : public BatchDataset<S, Tensor> {
 public:
  using ExampleType = Example<>;

  virtual Example<> index(size_t index) = 0;

  Example<> batch(ArrayRef<size_t> indices) override {
    std::vector<Tensor> data(indices.size());
    std::vector<Tensor> labels(indices.size());
    for (const auto i : indices) {
      auto example = index(i);
      data[i] = std::move(example.data);
      labels[i] = std::move(example.label);
    }
    return {torch::stack(data), torch::stack(labels)};
  }
};

// Dataset specialization for Tensor; collates a batch of tensors into a
// single tensor.
template <typename S>
class Dataset<S, Tensor, Tensor> : public BatchDataset<S, Tensor> {
 public:
  using ExampleType = Tensor;

  virtual Tensor index(size_t index) = 0;

  Tensor batch(ArrayRef<size_t> indices) override {
    return torch::stack(
        torch::fmap(indices, [this](size_t i) { return this->index(i); }));
  }
};

} // namespace datasets
} // namespace data
} // namespace torch

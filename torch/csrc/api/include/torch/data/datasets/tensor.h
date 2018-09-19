#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/tensor.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace data {
namespace datasets {
class TensorDataset : public Dataset<TensorDataset, TensorExample> {
 public:
  explicit TensorDataset(std::vector<Tensor> tensors)
      : tensors_(std::move(tensors)) {}

  TensorExample index(size_t index) override {
    return tensors_[index];
  }

  size_t size() const override {
    return tensors_.size();
  }

 private:
  std::vector<Tensor> tensors_;
};

} // namespace datasets
} // namespace data
} // namespace torch

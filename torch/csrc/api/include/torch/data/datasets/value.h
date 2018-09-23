#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/tensor.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace data {
namespace datasets {

template <typename E, typename B = std::vector<E>>
class ValueDataset : public Dataset<ValueDataset<E>, E, B> {
 public:
  using ExampleType = E;

  explicit ValueDataset(ArrayRef<E> values)
      : values_(values.begin(), values.end()) {}

  ExampleType index(size_t index) override {
    return values_[index];
  }

  size_t size() const override {
    return values_.size();
  }

 private:
  std::vector<E> values_;
};

using TensorDataset = ValueDataset<Tensor, Tensor>;
} // namespace datasets
} // namespace data
} // namespace torch

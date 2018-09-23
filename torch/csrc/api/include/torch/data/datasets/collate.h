#pragma once

#include <torch/data/example.h>
#include <torch/data/transforms/base.h>
#include <torch/tensor.h>

#include <functional>
#include <utility>
#include <vector>

namespace torch {
namespace data {
namespace transforms {

// typename I, typename O, not just E
template <typename E>
class Collate : public BatchTransform<std::vector<E>, E> {
 public:
  using CollationFunction = std::function<E(std::vector<E>)>;

  explicit Collate(CollationFunction function)
      : function_(std::move(function)) {}

  E apply_batch(std::vector<E> input_batch) {
    return function_(std::move(input_batch));
  }

 private:
  CollationFunction function_;
};

template <typename E = Example<>>
struct Stack;

template <>
struct Stack<Example<>>
    : public BatchTransform<std::vector<Example<>>, Example<>> {
  Example<> apply_batch(std::vector<Example<>> examples) override {
    std::vector<torch::Tensor> data, labels;
    data.reserve(examples.size());
    labels.reserve(examples.size());
    for (auto& example : examples) {
      data.push_back(std::move(example.data));
      labels.push_back(std::move(example.label));
    }
    return {torch::stack(data), torch::stack(labels)};
  }
};

template <>
struct Stack<Tensor> : public BatchTransform<std::vector<Tensor>, Tensor> {
  Tensor apply_batch(std::vector<Tensor> tensors) override {
    return torch::stack(tensors);
  }
};

} // namespace transforms
} // namespace data
} // namespace torch

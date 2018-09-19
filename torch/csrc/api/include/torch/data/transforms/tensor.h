#pragma once

#include <torch/data/example.h>
#include <torch/data/transforms/base.h>
#include <torch/tensor.h>

#include <functional>
#include <utility>

namespace torch {
namespace data {
namespace transforms {

template <typename LabelType = Tensor>
class TensorTransform
    : public Transform<Example<Tensor, LabelType>, Example<Tensor, LabelType>> {
 public:
  using E = Example<Tensor, LabelType>;
  using typename Transform<E, E>::InputType;
  using typename Transform<E, E>::OutputType;

  virtual Tensor operator()(Tensor input) = 0;

  OutputType apply(InputType input) override {
    input.data = (*this)(std::move(input.data));
    return input;
  }
};

template <typename LabelType = Tensor>
struct Normalize : TensorTransform<LabelType> {
  Normalize(double mean, double stddev) : mean(mean), stddev(stddev) {}

  Tensor operator()(Tensor input) override {
    return (input - mean) / stddev;
  }

  double mean, stddev;
};

template <typename LabelType = Tensor>
struct Lambda : TensorTransform<LabelType> {
  using FunctionType = std::function<Tensor(Tensor)>;

  explicit Lambda(FunctionType function) : function(std::move(function)) {}

  Tensor operator()(Tensor input) override {
    return function(std::move(input));
  }

  FunctionType function;
};
} // namespace transforms
} // namespace data
} // namespace torch

#pragma once

#include <torch/tensor.h>

namespace torch {
namespace data {
namespace example {
using NoLabel = void;
} //  namespace example

template <typename D = Tensor, typename L = Tensor>
struct Example {
  using DataType = D;
  using LabelType = L;
  Example(DataType data, LabelType label)
      : data(std::move(data)), label(std::move(label)) {}
  DataType data;
  LabelType label;
};

template <typename D>
struct Example<D, example::NoLabel> {
  using DataType = D;
  using LabelType = example::NoLabel;

  /* implicit */ Example(DataType data) : data(std::move(data)) {}

  operator DataType&() {
    return data;
  }
  operator const DataType&() const {
    return data;
  }

  DataType data;
};

using TensorExample = Example<Tensor, example::NoLabel>;
} // namespace data
} // namespace torch

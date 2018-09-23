#pragma once

#include <torch/data/example.h>
#include <torch/detail/utils.h>
#include <torch/tensor.h>

#include <ATen/core/ArrayRef.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace torch {
namespace data {
namespace datasets {
template <typename S, typename T>
struct Map;
template <typename D, typename T>
Map<D, T> map(D&&, T&&);
} // namespace datasets
} // namespace data
} // namespace torch

namespace torch {
namespace data {
namespace datasets {

template <typename S, typename B = std::vector<Example<>>>
class BatchDataset {
 public:
  using Self = S;
  using BatchType = B;

  virtual ~BatchDataset() = default;

  virtual BatchType batch(ArrayRef<size_t> indices) = 0;
  virtual size_t size() const = 0;

  template <typename TransformType>
  Map<Self, TransformType> map(TransformType transform) && {
    return datasets::map(
        std::move(static_cast<Self&>(*this)), std::move(transform));
  }
};

template <typename S, typename E = Example<>, typename B = std::vector<E>>
class Dataset : public BatchDataset<S, B> {
 public:
  using typename BatchDataset<S, B>::BatchType;
  using ExampleType = E;

  virtual ExampleType index(size_t index) = 0;

  BatchType batch(ArrayRef<size_t> indices) override {
    BatchType batch;
    torch::detail::reserve_capacity(batch, indices.size());
    for (const auto i : indices) {
      batch.insert(batch.end(), index(i));
    }
    return batch;
  }
};
} // namespace datasets
} // namespace data
} // namespace torch

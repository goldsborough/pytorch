#pragma once

#include <torch/csrc/utils/functional.h>
#include <torch/data/example.h>
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

// The most basic dataset type. Maps from a batch of indices to some batch type.
template <typename S, typename B>
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

// A Dataset has also an ElementType and must provide a way of accessing
// individual elements besides entire batches. The batch type defaults to a
// vector of the element type. Note that this default case has a specialization
// below.
template <typename S, typename E = Tensor, typename B = std::vector<E>>
class Dataset : public BatchDataset<S, B> {
 public:
  using ExampleType = E;
  virtual ExampleType index(size_t index) = 0;
};

/// If the batch type is the default type (vector), we can provide a default
/// implementation.
template <typename S, typename E>
class Dataset<S, E, std::vector<E>> : public BatchDataset<S, std::vector<E>> {
 public:
  using ExampleType = E;
  virtual ExampleType index(size_t index) = 0;
  std::vector<E> batch(ArrayRef<size_t> indices) override {
    return torch::fmap(indices, [this](size_t i) { return this->index(i); });
  }
};

} // namespace datasets
} // namespace data
} // namespace torch

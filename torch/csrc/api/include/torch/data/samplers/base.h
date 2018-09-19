#pragma once

#include <ATen/optional.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace data {
namespace samplers {

template <typename I = std::vector<size_t>>
class Sampler {
 public:
  using IndexBatchType = I;
  virtual ~Sampler() = default;
  virtual void reset() = 0;
  virtual at::optional<IndexBatchType> next(size_t batch_size) = 0;
};

} // namespace samplers
} // namespace data
} // namespace torch

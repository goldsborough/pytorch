#pragma once

#include <torch/data/samplers/base.h>

#include <ATen/optional.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

namespace torch {
namespace data {
namespace samplers {
template <typename IndexType = uint32_t, typename RNG = std::mt19937>
class RandomSampler : public Sampler<> {
 public:
  RandomSampler(size_t size, RNG rng) : indices_(size), rng_(std::move(rng)) {
    std::iota(indices_.begin(), indices_.end(), 0);
    reset();
  }

  explicit RandomSampler(size_t size, IndexType seed = std::random_device{}())
      : RandomSampler(size, RNG(seed)) {}

  void reset() override {
    std::shuffle(indices_.begin(), indices_.end(), std::move(rng_));
    index_ = 0;
  }

  at::optional<std::vector<size_t>> next(size_t batch_size) override {
    const auto remaining_indices = indices_.size() - index_;
    if (remaining_indices == 0) {
      return at::nullopt;
    }
    std::vector<size_t> index_batch(std::min(batch_size, remaining_indices));
    for (size_t i = 0; i < index_batch.size(); ++i) {
      index_batch[i] = indices_[index_++];
    }
    return index_batch;
  }

 private:
  std::vector<IndexType> indices_;
  size_t index_{0};
  RNG rng_;
};
} // namespace samplers
} // namespace data
} // namespace torch

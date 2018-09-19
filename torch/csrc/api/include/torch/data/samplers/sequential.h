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

class SequentialSampler : public Sampler<> {
 public:
  SequentialSampler(size_t dataset_size) : dataset_size_(dataset_size) {}

  void reset() override {
    index_ = 0;
  }

  at::optional<std::vector<size_t>> next(size_t batch_size) override {
    const auto remaining_indices = dataset_size_ - index_;
    if (remaining_indices == 0) {
      return at::nullopt;
    }
    std::vector<size_t> index_batch(std::min(batch_size, remaining_indices));
    for (size_t i = 0; i < index_batch.size(); ++i) {
      index_batch[i] = index_++;
    }
    return index_batch;
  }

 private:
  size_t dataset_size_;
  size_t index_{0};
};

} // namespace samplers
} // namespace data
} // namespace torch

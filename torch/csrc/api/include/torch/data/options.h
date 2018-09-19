#pragma once

#include <torch/arg.h>

#include <ATen/optional.h>

#include <chrono>
#include <cstddef>

namespace torch {
namespace data {
struct DataLoaderOptions {
  DataLoaderOptions() = default;

  DataLoaderOptions& coalesce() {
    if (!max_jobs_.has_value()) {
      max_jobs_ = 2 * workers_;
    }
    return *this;
  }

  TORCH_ARG(size_t, batch_size) = 1;
  TORCH_ARG(bool, drop_last) = false;
  TORCH_ARG(size_t, workers) = 0;
  TORCH_ARG(at::optional<size_t>, max_jobs);
  TORCH_ARG(at::optional<std::chrono::seconds>, timeout);
  TORCH_ARG(bool, enforce_ordering) = false;
  TORCH_ARG(bool, defer_prefetch) = false;
};
} // namespace data
} // namespace torch

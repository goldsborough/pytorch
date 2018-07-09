#pragma once

#include <torch/nn/module.h>
#include <torch/nn/modules/any.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <ATen/Backtrace.h>
#include <ATen/Error.h>
#include <ATen/optional.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace torch {
namespace nn {

class DataParallelImpl : public torch::nn::Module {
 public:
  explicit DataParallelImpl(
      AnyModule module,
      at::optional<std::vector<Device>> devices = at::nullopt,
      at::optional<Device> output_device = at::nullopt,
      int64_t dim = 0);

  Tensor forward(Tensor input);

  ArrayRef<Device> devices() const noexcept;
  const Device& output_device() const noexcept;
  int64_t dim() const noexcept;

 private:
  AnyModule module_;
  std::vector<Device> devices_;
  Device output_device_;
  int64_t dim_;
};

TORCH_MODULE(DataParallel);

} // namespace nn
} // namespace torch

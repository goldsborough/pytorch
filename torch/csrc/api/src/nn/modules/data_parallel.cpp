#include <torch/nn/modules/data_parallel.h>

#include <torch/cuda.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/any.h>
#include <torch/nn/parallel/data_parallel.h>
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
namespace {
std::vector<AnyModule> replicate(
    const AnyModule& module,
    const std::vector<Device>& devices) {
  std::vector<AnyModule> replicas;
  replicas.reserve(devices.size());
  for (const auto& device : devices) {
    OptionsGuard guard(device);
    replicas.push_back(module->clone());
  }
  return replicas;
}

std::vector<Tensor> convert_any_to_tensor(
    std::vector<AnyModule::Value>&& anys) {
  std::vector<Tensor> tensors;
  tensors.reserve(anys.size());
  for (auto& tensor : anys) {
    tensors.push_back(tensor.get<Tensor>());
  }
  return tensors;
}
} // namespace

DataParallelImpl::DataParallelImpl(
    AnyModule module,
    at::optional<std::vector<Device>> devices,
    at::optional<Device> output_device)
    : module_(std::move(module)),
      devices_(std::move(devices).value_or(
          parallel::detail::all_available_devices())),
      output_device_(std::move(output_device).value_or(devices_.front())) {}

Tensor DataParallelImpl::forward(Tensor input) {
  autograd::Scatter scatter(devices_, /*chunk_sizes=*/at::nullopt, dim_);
  auto scattered_inputs = scatter.apply({input});

  if (devices_.size() == 1) {
    return module_.forward(input);
  }

  auto replicas = replicate(module_, devices_);
  auto any_outputs = parallel::parallel_apply<AnyModule, AnyModule::Value>(
      replicas, scattered_inputs, devices_);
  auto outputs = convert_any_to_tensor(std::move(any_outputs));

  return autograd::Gather(output_device_, dim).apply(outputs).front();
}

ArrayRef<Device> DataParallelImpl::devices() const noexcept {
  return devices_;
}
const Device& DataParallelImpl::output_device() const noexcept {
  return output_device_;
}

int64_t DataParallelImpl::dim() const noexcept {
  return dim_;
}

} // namespace nn
} // namespace torch

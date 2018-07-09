#include <torch/csrc/autograd/functions/comm.h>

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/cuda/comm.h>
#include <torch/csrc/utils/functional.h>

#include <ATen/ATen.h>
#include <ATen/optional.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace torch {
namespace autograd {
Scatter::Scatter(
    std::vector<at::Device> devices,
    const at::optional<std::vector<int64_t>>& chunk_sizes,
    int64_t dim,
    const at::optional<std::vector<THCStream*>>& streams,
    bool unsqueeze_scalars)
    : devices_(std::move(devices)),
      chunk_sizes_(chunk_sizes),
      dim_(dim),
      streams_(streams),
      unsqueeze_scalars_(unsqueeze_scalars) {}

variable_list Scatter::apply(const variable_list& inputs) {
  AT_ASSERT(inputs.size() == 1);
  auto& input = inputs.front();

  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad(input)) {
    grad_fn =
        std::make_shared<Gather>(/*destination_device=*/input.device(), dim_);
    grad_fn->set_next_edges(collect_next_edges(input));
  }

  auto device_indices = fmap(devices_, [](const at::Device& device) -> int64_t {
    return device.index();
  });
  std::vector<at::Tensor> tensors;
#ifdef USE_CUDA
  tensors =
      torch::cuda::scatter(input, device_indices, chunk_sizes_, dim_, streams_);
#else
  AT_ERROR("Scatter is only supported in CUDA environments");
#endif

  std::vector<Variable> variables;
  for (auto& tensor : tensors) {
    AT_ASSERT(tensor.defined());
    if (unsqueeze_scalars_) {
      AT_ASSERT(tensor.dim() == 1 && tensor.numel() == 1);
      variables.push_back(tensor[0]);
    } else {
      variables.push_back(tensor);
    }
  }

  set_history(variables, grad_fn);

  return variables;
}

Gather::Gather(const at::Device& destination_device, int64_t dim)
    : destination_device_(destination_device), dim_(dim) {}

variable_list Gather::apply(const variable_list& inputs) {
  bool all_are_zero_dim = true;
  for (auto& input : inputs) {
    AT_CHECK(
        input.is_cuda(),
        "All inputs to Gather must be CUDA tensors, got ",
        input.type());
    if (input.dim() > 0) {
      all_are_zero_dim = false;
    }
  }

  bool unsqueeze_scalars = all_are_zero_dim && dim_ == 0;
  if (unsqueeze_scalars) {
    AT_WARN(
        "Was asked to gather along dimension 0, but all "
        "input tensors were scalars; will instead unsqueeze "
        "and return a vector.");
  }

  std::vector<at::Tensor> tensors;
  for (auto& variable : inputs) {
    if (unsqueeze_scalars) {
      tensors.push_back(variable.view(1));
    } else {
      tensors.push_back(variable);
    }
  }

  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad(inputs)) {
    std::vector<at::Device> source_devices;
    std::vector<int64_t> input_sizes;
    for (auto& input : inputs) {
      source_devices.push_back(input.device());
      input_sizes.push_back(input.size(dim_));
    }
    grad_fn = std::make_shared<Scatter>(
        source_devices,
        input_sizes,
        dim_,
        /*streams=*/at::nullopt,
        /*unsqueeze_scalars=*/unsqueeze_scalars);
    grad_fn->set_next_edges(collect_next_edges(inputs));
  }

  Variable variable;
#ifdef USE_CUDA
  // This is special logic for torch::cuda::gather!
  const auto destination_index =
      destination_device_.is_cpu() ? -1 : destination_device_.index();
  variable = torch::cuda::gather(tensors, dim_, destination_index);
#else
  AT_ERROR("Gather is only supported in CUDA environments");
#endif

  set_history(variable, grad_fn);

  return {variable};
}

} // namespace autograd
} // namespace torch

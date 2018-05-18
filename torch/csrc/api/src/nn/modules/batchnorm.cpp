#include <torch/nn/modules/batchnorm.h>

#include <cstdint>

namespace torch { namespace nn {

BatchNorm::BatchNorm(int64_t features) : features_(features) {}

void BatchNorm::reset() {
  if (affine_) {
    register_parameter(
        "weight",
        &BatchNorm::weight_,
        at::CPU(at::kFloat).empty({features_}).uniform_());
    register_parameter(
        "bias", &BatchNorm::bias_, at::CPU(at::kFloat).zeros({features_}));
  }

  if (stateful_) {
    // TODO: Make into buffers instead of parameters
    // running_mean_ = add(
    //     Var(at::CPU(at::kFloat).zeros({features_}), false), "running_mean_");
    // running_variance_ = add(
    //     Var(at::CPU(at::kFloat).ones({features_}), false), "running_variance_");
  }
}

variable_list BatchNorm::forward(variable_list inputs) {
  auto& input = inputs[0];
  auto& running_mean_ = (stateful_ ? this->running_mean_ : inputs[1]);
  auto& running_variance_ = (stateful_ ? this->running_variance_ : inputs[2]);

  if (is_training()) {
    const auto num_channels = input.dim() > 1 ? input.size(1) : 1;
    if (input.numel() / num_channels <= 1) {
      throw std::runtime_error(
          "BatchNorm expected more than 1 value per channel when training!");
    }
  }

  auto output = at::batch_norm(
      input,
      weight_,
      bias_,
      running_mean_,
      running_variance_,
      is_training(),
      momentum_,
      eps_,
      hasCudnn());

  return variable_list({output});
}
}} // namespace torch::nn

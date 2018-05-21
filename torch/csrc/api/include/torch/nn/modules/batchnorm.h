#pragma once

#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

#include <cstdint>

namespace torch { namespace nn {
class BatchNorm : public torch::nn::CloneableModule<BatchNorm> {
 public:
  explicit BatchNorm(int64_t features);

  void reset() override;

  variable_list forward(variable_list) override;

  TORCH_ARG(int64_t, features);
  TORCH_ARG(bool, affine) = true;
  TORCH_ARG(bool, stateful) = false;
  TORCH_ARG(double, eps) = 1e-5;
  TORCH_ARG(double, momentum) = 0.1;
  TORCH_ARG(Variable, weight);
  TORCH_ARG(Variable, bias);
  TORCH_ARG(Variable, running_mean);
  TORCH_ARG(Variable, running_variance);
};
}} // namespace torch::nn

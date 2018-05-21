#pragma once

#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

#include <cstdint>

namespace torch {
namespace nn {

struct LinearOptions {
  LinearOptions(int64_t in, int64_t out);
  TORCH_ARG(int64_t, in);
  TORCH_ARG(int64_t, out);
  TORCH_ARG(bool, with_bias) = true;
};

class LinearImpl : public torch::nn::Module {
 public:
  explicit LinearImpl(LinearOptions options);

  variable_list forward(variable_list input) override;

  const LinearOptions& options() const noexcept;

 private:
  Variable weight_;
  Variable bias_;
  LinearOptions options_;
};

TORCH_MODULE(Linear);

} // namespace nn
} // namespace torch

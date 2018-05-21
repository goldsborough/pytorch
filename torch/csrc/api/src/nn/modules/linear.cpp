#include <torch/nn/modules/linear.h>

#include <ATen/ATen.h>

#include <cmath>
#include <cstdint>

namespace torch {
namespace nn {

LinearOptions::LinearOptions(int64_t in, int64_t out) : in_(in), out_(out) {}

LinearImpl::LinearImpl(LinearOptions options) : options_(std::move(options)) {
  weight_ = add(
      Var(at::CPU(at::kFloat).empty({options_.out_, options_.in_})), "weight");
  if (options_.with_bias_) {
    bias_ = add(Var(at::CPU(at::kFloat).empty(options_.out_)), "bias");
  }

  const auto stdv = 1.0 / std::sqrt(weight_.size(1));
  for (auto& p : parameters()) {
    p.second.data().uniform_(-stdv, stdv);
  }
}

variable_list LinearImpl::forward(variable_list input) {
  auto x = input[0];
  if (x.ndimension() == 2 && options_.with_bias_) {
    // Fused op is marginally faster
    AT_ASSERT(x.size(1) == weight_.size(1));
    return variable_list({at::addmm(bias_, x, weight_.t())});
  }

  auto output = x.matmul(weight_.t());
  if (options_.with_bias_) {
    output += bias_;
  }
  return variable_list({output});
}

const LinearOptions& LinearImpl::options() const noexcept {
  return options_;
}

} // namespace nn
} // namespace torch

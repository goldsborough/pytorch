#pragma once

#include <torch/expanding_array.h>
#include <torch/nn/module.h>

#include <cstdint>

namespace torch {
namespace nn {

template <size_t D>
struct ConvOptions {
  ConvOptions(
      int64_t input_channels,
      int64_t output_channels,
      ExpandingArray<D> kernel_size);

  TORCH_ARG(int64_t, input_channels);
  TORCH_ARG(int64_t, output_channels);
  TORCH_ARG(ExpandingArray<D>, kernel_size);
  TORCH_ARG(ExpandingArray<D>, stride) = 1;
  TORCH_ARG(ExpandingArray<D>, padding) = 0;
  TORCH_ARG(ExpandingArray<D>, dilation) = 1;
  TORCH_ARG(ExpandingArray<D>, output_padding) = 0;
  TORCH_ARG(bool, transposed) = false;
  TORCH_ARG(bool, with_bias) = true;
  TORCH_ARG(int64_t, groups) = 1;
};

template <size_t D, typename Derived>
class ConvImpl : public torch::nn::Module {
 public:
  explicit ConvImpl(ConvOptions<D> options);

  const ConvOptions<D>& options() const noexcept;

 protected:
  Variable weight_;
  Variable bias_;
  ConvOptions<D> options_;
};

#define CONV_D(D)                                             \
  class Conv##D##dImpl : public ConvImpl<D, Conv##D##dImpl> { \
   public:                                                    \
    using ConvImpl<D, Conv##D##dImpl>::ConvImpl;              \
    variable_list forward(variable_list) override;            \
  };                                                          \
  using Conv##D##dOptions = ConvOptions<D>;                   \
  TORCH_MODULE(Conv##D##d)

CONV_D(1);
CONV_D(2);
CONV_D(3);

#undef CONV_D

} // namespace nn
} // namespace torch

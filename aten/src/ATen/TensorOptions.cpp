#include <ATen/TensorOptions.h>

#include <ATen/DefaultTensorOptions.h>

namespace at {
TensorOptions::TensorOptions() {
  auto default_options = DefaultTensorOptions::copy();
  this->dtype(default_options.dtype());
  this->device(default_options.device());
  this->layout(default_options.layout());
}
} // namespace at

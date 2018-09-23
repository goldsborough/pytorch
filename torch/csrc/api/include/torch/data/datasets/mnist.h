#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>

#include <cstddef>
#include <string>

namespace torch {
namespace data {
namespace datasets {
class MNIST : public Dataset<MNIST> {
 public:
  explicit MNIST(const std::string& root, bool train = true);

  Example<> index(size_t index) override;
  size_t size() const override;
  bool is_train() const noexcept;

 private:
  Tensor images_;
  Tensor labels_;
};
} // namespace datasets
} // namespace data
} // namespace torch

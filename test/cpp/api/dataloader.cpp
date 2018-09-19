#include <gtest/gtest.h>

#include <torch/data.h>
#include <torch/tensor.h>

#include <chrono>
#include <iostream>
#include <vector>

using namespace torch::data;

TEST(TestDataLoader, TestA) {
  auto dataset = datasets::MNIST("test/cpp/api/mnist")
                     .map(transforms::Lambda<>(
                         [](torch::Tensor input) { return input.mul(10); }))
                     .map(transforms::Normalize<>(0, 1))
                     .map(transforms::Normalize<>(0, 1))
                     .map(transforms::Normalize<>(0, 1));

  auto data_loader = torch::data::data_loader(
      std::move(dataset),
      DataLoaderOptions().batch_size(4).workers(2).enforce_ordering(true));

  for (size_t epoch = 0; epoch < 2; ++epoch) {
    std::cout << "==================================== epoch: " << epoch
              << std::endl;
    while (auto batch = data_loader->next()) {
      std::cout << "batch of " << batch->size() << std::endl;
      for (const auto& example : *batch) {
        std::cout << example.label << " -> " << example.data << std::endl;
      }
      std::cout << "-----------------------------------------" << std::endl;
    }
    data_loader->reset();
  }
}

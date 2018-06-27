#include <torch/torch.h>

#include <iostream>
#include <vector>

using namespace torch;

struct Reshape : nn::Module {
  explicit Reshape(std::vector<int64_t> shape) : shape_(shape) {}
  torch::Tensor forward(torch::Tensor input) {
    std::vector<int64_t> new_shape = {input.size(0)};
    new_shape.insert(new_shape.end(), shape_.begin(), shape_.end());
    return input.view(new_shape);
  }
  std::vector<int64_t> shape_;
};

struct AddNoise : nn::Module {
  explicit AddNoise(double mean = 0.0, double stddev = 0.1)
      : mean_(mean), stddev_(stddev) {}

  torch::Tensor forward(torch::Tensor input) {
    auto noise = torch::empty_like(input).normal_(mean_, stddev_);
    return input + noise;
  }

  double mean_;
  double stddev_;
};

auto main() -> int {
  const int64_t kNoiseSize = 100;
  const int64_t kNumberOfEpochs = 30;
  const int64_t kBatchSize = 256;
  const bool kUseCUDA = true;

  nn::Sequential generator(
      // Layer 1
      nn::Linear(kNoiseSize, 7 * 7 * 256),
      nn::Functional(at::leaky_relu, /*negative_slope=*/0.2),
      Reshape({256, 7, 7}),
      // Layer 2
      nn::Functional(at::upsample_nearest2d, /*scale_factor=*/2),
      nn::Conv2d(nn::Conv2dOptions(256, 128, /*kernel_size=*/5).padding(2)),
      nn::BatchNorm(nn::BatchNormOptions(128).momentum(0.9)),
      nn::Functional(at::leaky_relu, /*negative_slope=*/0.2),
      // Layer 3
      nn::Functional(at::upsample_nearest2d, /*scale_factor=*/2),
      nn::Conv2d(nn::Conv2dOptions(128, 64, /*kernel_size=*/5).padding(2)),
      nn::BatchNorm(nn::BatchNormOptions(64).momentum(0.9)),
      nn::Functional(at::leaky_relu, /*negative_slope=*/0.2),
      // Layer 4
      nn::Functional(at::upsample_nearest2d, /*scale_factor=*/2),
      nn::Conv2d(nn::Conv2dOptions(64, 32, /*kernel_size=*/5).padding(2)),
      nn::BatchNorm(nn::BatchNormOptions(32).momentum(0.9)),
      nn::Functional(at::leaky_relu, /*negative_slope=*/0.2),
      // Output
      nn::Conv2d(nn::Conv2dOptions(32, 1, /*kernel_size=*/5).padding(2)),
      nn::Functional(at::tanh));

  generator.parameters().apply_items(initialize_weights);

  nn::Sequential discriminator(
      AddNoise(),
      // Layer 1
      nn::Conv2d(
          nn::Conv2dOptions(1, 32, /*kernel_size=*/5).padding(2).stride(2)),
      nn::Functional(at::leaky_relu, /*negative_slope=*/0.2),
      // Layer 2
      nn::Conv2d(
          nn::Conv2dOptions(32, 64, /*kernel_size=*/5).padding(2).stride(2)),
      nn::Functional(at::leaky_relu, /*negative_slope=*/0.2),
      // Layer 3
      nn::Conv2d(
          nn::Conv2dOptions(64, 128, /*kernel_size=*/5).padding(2).stride(2)),
      nn::Functional(at::leaky_relu, /*negative_slope=*/0.2),
      // Layer 4
      nn::Conv2d(
          nn::Conv2dOptions(128, 256, /*kernel_size=*/5).padding(2).stride(2)),
      nn::Functional(at::leaky_relu, /*negative_slope=*/0.2),
      // Output
      Reshape({-1}),
      nn::Linear(256 * 2 * 2, 1));

  optim::Adam generator_optimizer(
      generator.parameters(), optim::AdamOptions(2e-4).beta1(0.5));
  optim::Adam discriminator_optimizer(
      discriminator.parameters(), optim::AdamOptions(5e-4).beta1(0.5));

  for (size_t epoch = 0; epoch < kNumberOfEpochs; ++epoch) {

  }

  std::cout << generator.forward(torch::randn({1, kNoiseSize})) << std::endl;
}

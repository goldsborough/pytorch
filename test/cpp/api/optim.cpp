#include <catch.hpp>

#include <torch/nn/module.h>
#include <torch/nn/modules/functional.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/sequential.h>
#include <torch/optim.h>
#include <torch/tensor.h>
#include <torch/utils.h>

#include <test/cpp/api/optim_baseline.h>
#include <test/cpp/api/util.h>

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

using namespace torch::nn;
using namespace torch::optim;

bool test_optimizer_xor(Optimizer&& optimizer, Sequential& model) {
  const int64_t kBatchSize = 4;

  float running_loss = 1;
  int epoch = 0;
  while (running_loss > 0.1) {
    auto inputs = torch::empty({kBatchSize, 2});
    auto labels = torch::empty({kBatchSize});
    for (size_t i = 0; i < kBatchSize; i++) {
      inputs[i] = torch::randint(2, {2}, torch::kInt64);
      labels[i] = inputs[i][0].toCLong() ^ inputs[i][1].toCLong();
    }
    inputs.set_requires_grad(true);
    optimizer.zero_grad();
    auto x = model.forward(inputs);
    torch::Tensor loss = torch::binary_cross_entropy(x, labels);
    loss.backward();

    optimizer.step();

    running_loss = running_loss * 0.99 + loss.toCFloat() * 0.01;
    if (epoch > 3000) {
      return false;
    }
    epoch++;
  }
  return true;
}

template <typename Parameters>
void assign_parameter(
    const Parameters& parameters,
    const char* name,
    torch::Tensor new_tensor) {
  auto parameter = parameters.at(name);
  parameter.set_requires_grad(false);
  parameter.flatten().copy_(new_tensor);
  parameter.set_requires_grad(true);
}

template <typename OptimizerClass, typename Options>
void check_exact_values(
    Options options,
    std::vector<std::vector<torch::Tensor>> expected_parameters) {
  const size_t kIterations = 1001;
  const size_t kSampleEvery = 100;

  Sequential model(
      Linear(2, 3),
      Functional(torch::sigmoid),
      Linear(3, 1),
      Functional(torch::sigmoid));

  model.to(torch::kFloat64);

  // Use exact input values because matching random values is hard.
  auto parameters = model.parameters();
  assign_parameter(
      parameters,
      "0.weight",
      torch::tensor({-0.2109, -0.4976, -0.1413, -0.3420, -0.2524, 0.6976}));
  assign_parameter(
      parameters, "0.bias", torch::tensor({-0.1085, -0.2979, 0.6892}));
  assign_parameter(
      parameters, "2.weight", torch::tensor({-0.0508, -0.3941, -0.2843}));
  assign_parameter(parameters, "2.bias", torch::tensor({-0.0711}));

  auto optimizer = OptimizerClass(parameters, options);
  torch::Tensor input =
      torch::tensor({0.1, 0.2, 0.3, 0.4, 0.5, 0.6}).reshape({3, 2});

  for (size_t i = 0; i < kIterations; ++i) {
    optimizer.zero_grad();
    auto output = model.forward(input);
    auto loss = output.sum();
    loss.backward();

    optimizer.step();

    if (i % kSampleEvery == 0) {
      REQUIRE(
          expected_parameters.at(i / kSampleEvery).size() == parameters.size());
      for (size_t p = 0; p < parameters.size(); ++p) {
        REQUIRE(parameters.at(p)->defined());
        auto computed = parameters.at(p)->flatten();
        auto expected = expected_parameters.at(i / kSampleEvery).at(p);
        if (!computed.allclose(expected, /*rtol=*/1e-3, /*atol=*/1e-5)) {
          std::cout << "Iteration " << i << ": " << computed
                    << " != " << expected << " (parameter " << p << ")"
                    << std::endl;
          REQUIRE(false);
        }
      }
    }
  }
}

TEST_CASE("Optim/XORConvergence") {
  Sequential model(
      Linear(2, 8),
      Functional(torch::sigmoid),
      Linear(8, 1),
      Functional(torch::sigmoid));

  SECTION("sgd") {
    REQUIRE(test_optimizer_xor(
        SGD(model.parameters(),
            SGDOptions(1e-1).momentum(0.9).nesterov(true).weight_decay(1e-6)),
        model));
  }

  SECTION("adagrad") {
    REQUIRE(test_optimizer_xor(
        Adagrad(
            model.parameters(),
            AdagradOptions(1.0).weight_decay(1e-6).lr_decay(1e-3)),
        model));
  }

  SECTION("rmsprop_simple") {
    REQUIRE(test_optimizer_xor(
        RMSprop(model.parameters(), RMSpropOptions(1e-1).centered(true)),
        model));
  }

  SECTION("rmsprop") {
    REQUIRE(test_optimizer_xor(
        RMSprop(
            model.parameters(),
            RMSpropOptions(1e-1).momentum(0.9).weight_decay(1e-6)),
        model));
  }

  SECTION("adam") {
    REQUIRE(test_optimizer_xor(
        Adam(model.parameters(), AdamOptions(1.0).weight_decay(1e-6)), model));
  }

  SECTION("amsgrad") {
    REQUIRE(test_optimizer_xor(
        Adam(
            model.parameters(),
            AdamOptions(0.1).weight_decay(1e-6).amsgrad(true)),
        model));
  }
}

TEST_CASE("Optim/ProducesPyTorchValues/Adam") {
  check_exact_values<Adam>(
      AdamOptions(1.0).weight_decay(1e-6), expected_parameters::Adam);
}

TEST_CASE("Optim/ProducesPyTorchValues/Adagrad") {
  check_exact_values<Adagrad>(
      AdagradOptions(1.0).weight_decay(1e-6).lr_decay(1e-3),
      expected_parameters::Adagrad);
}

TEST_CASE("Optim/ProducesPyTorchValues/RMSprop") {
  check_exact_values<RMSprop>(
      RMSpropOptions(1e-1).momentum(0.9).weight_decay(1e-6),
      expected_parameters::RMSprop);
}

TEST_CASE("Optim/ProducesPyTorchValues/SGD") {
  check_exact_values<SGD>(
      SGDOptions(1e-1).momentum(0.9).weight_decay(1e-6),
      expected_parameters::SGD);
}

TEST_CASE("Optim/ZeroGrad") {
  Linear model(2, 8);
  SGD optimizer(model->parameters(), 0.1);

  for (const auto& parameter : model->parameters()) {
    REQUIRE(!parameter->grad().defined());
  }

  auto output = model->forward(torch::ones({5, 2}));
  auto loss = output.sum();
  loss.backward();

  for (const auto& parameter : model->parameters()) {
    REQUIRE(parameter->grad().defined());
    REQUIRE(parameter->grad().sum().toCFloat() > 0);
  }

  optimizer.zero_grad();

  for (const auto& parameter : model->parameters()) {
    REQUIRE(parameter->grad().defined());
    REQUIRE(parameter->grad().sum().toCFloat() == 0);
  }
}

TEST_CASE("Optim/ExternalVectorOfParameters") {
  std::vector<torch::Tensor> parameters = {
      torch::randn({2, 2}), torch::randn({3, 3}), torch::randn({4, 4})};
  std::vector<torch::Tensor> original_parameters = {
      parameters[0].clone(), parameters[1].clone(), parameters[2].clone()};

  // Set all gradients to one
  for (auto& parameter : parameters) {
    parameter.grad() = torch::ones_like(parameter);
  }

  SGD optimizer(parameters, 1.0);

  optimizer.step();

  REQUIRE(parameters[0].allclose(original_parameters[0] - 1.0));
  REQUIRE(parameters[1].allclose(original_parameters[1] - 1.0));
  REQUIRE(parameters[2].allclose(original_parameters[2] - 1.0));
}

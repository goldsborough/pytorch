#include <catch.hpp>

#include <torch/csrc/autograd/functions/comm.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/parallel/data_parallel.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

using namespace torch::autograd;
using namespace torch::nn;

TEST_CASE("Parallel/DifferentiableScatter", "[cuda]") {
  Scatter scatter(
      {torch::Device(torch::kCUDA, 0), torch::Device(torch::kCUDA, 1)});

  auto input = torch::ones(10, torch::requires_grad(true));
  auto output = scatter.apply({input});

  REQUIRE(output.size() == 2);
  REQUIRE(output[0].size(0) == 5);
  REQUIRE(output[1].size(0) == 5);

  auto sum = output[0].to({torch::kCUDA, 1}) + output[1];
  sum.backward();

  REQUIRE(input.grad().defined());
  REQUIRE(input.grad().device().is_cpu());
  REQUIRE(input.grad().sum().toCInt() == 10);
}

TEST_CASE("Parallel/DifferentiableGather", "[cuda]") {
  Gather gather(torch::Device(torch::kCUDA, 1));

  auto a = torch::ones(5, torch::requires_grad(true).device({torch::kCUDA, 0}));
  auto b = torch::ones(5, torch::requires_grad(true).device({torch::kCUDA, 1}));
  auto output = gather.apply({a, b});

  REQUIRE(output.size() == 1);
  REQUIRE(output[0].size(0) == 10);
  REQUIRE(output[0].device() == torch::Device(torch::kCUDA, 1));

  output[0].backward();

  REQUIRE(a.grad().defined());
  REQUIRE(a.grad().device() == torch::Device(torch::kCUDA, 0));
  REQUIRE(a.grad().sum().toCInt() == 5);

  REQUIRE(b.grad().defined());
  REQUIRE(b.grad().device() == torch::Device(torch::kCUDA, 1));
  REQUIRE(b.grad().sum().toCInt() == 5);
}

TEST_CASE("Parallel/Replicate", "[cuda]") {
  Linear linear(3, 4);
  auto replicas = parallel::replicate(
      linear, {torch::Device(torch::kCUDA, 0), torch::Device(torch::kCUDA, 1)});
  REQUIRE(replicas.size() == 2);

  auto original_parameters = linear->parameters();

  auto replica1_parameters = replicas[0]->parameters();
  for (auto& parameter : replica1_parameters) {
    REQUIRE(parameter->device() == torch::Device(torch::kCUDA, 0));
  }
  replicas[0]->to(torch::kCPU);
  REQUIRE(replica1_parameters.size() == original_parameters.size());
  for (size_t i = 0; i < original_parameters.size(); ++i) {
    REQUIRE(replica1_parameters[i]->allclose(*original_parameters[i]));
    REQUIRE(
        replica1_parameters[i]->data().data<float>() !=
        original_parameters[i]->data().data<float>());
  }

  auto replica2_parameters = replicas[1]->parameters();
  for (auto& parameter : replica2_parameters) {
    REQUIRE(parameter->device() == torch::Device(torch::kCUDA, 1));
  }
  replicas[1]->to(torch::kCPU);
  REQUIRE(replica2_parameters.size() == original_parameters.size());
  for (size_t i = 0; i < original_parameters.size(); ++i) {
    REQUIRE(replica2_parameters[i]->allclose(*original_parameters[i]));
    REQUIRE(
        replica2_parameters[i]->data().data<float>() !=
        original_parameters[i]->data().data<float>());
  }
}

TEST_CASE("Parallel/ParallelApply", "[cuda]") {
  Linear a(3, 4);

  Linear b(std::static_pointer_cast<LinearImpl>(a->clone()));
  b->to({torch::kCUDA, 0});

  Linear c(std::static_pointer_cast<LinearImpl>(a->clone()));
  c->to({torch::kCUDA, 1});

  std::vector<Linear> modules = {a, b, c};
  std::vector<torch::Tensor> inputs = {
      torch::ones({2, 3}),
      torch::ones({2, 3}, torch::device({torch::kCUDA, 0})),
      torch::ones({2, 3}, torch::device({torch::kCUDA, 1}))};

  auto outputs = parallel::parallel_apply(modules, inputs);

  REQUIRE(outputs.size() == 3);
  REQUIRE(outputs[0].device().is_cpu());

  REQUIRE(outputs[1].device() == torch::Device(torch::kCUDA, 0));
  REQUIRE(outputs[1].to(torch::kCPU).allclose(outputs[0]));

  REQUIRE(outputs[2].device() == torch::Device(torch::kCUDA, 1));
  REQUIRE(outputs[2].to(torch::kCPU).allclose(outputs[0]));
}

TEST_CASE("Parallel/ParallelApplyWithDifferentOutputDevice", "[cuda]") {
  struct M : torch::nn::Module {
    torch::Tensor forward(torch::Tensor input) {
      return torch::ones({5}, torch::dtype(torch::kInt32));
    }
  };

  std::vector<std::shared_ptr<M>> modules = {
      std::make_shared<M>(), std::make_shared<M>(), std::make_shared<M>()};
  std::vector<torch::Tensor> inputs = {
      torch::empty({}), torch::empty({}), torch::empty({})};
  std::vector<torch::Device> devices = {
      {torch::kCUDA, 1}, {torch::kCUDA, 0}, {torch::kCPU}};

  auto outputs = parallel::parallel_apply(modules, inputs, devices);

  REQUIRE(outputs.size() == 3);
  REQUIRE(outputs[0].device().is_cuda());
  REQUIRE(outputs[0].device() == torch::Device(torch::kCUDA, 1));

  REQUIRE(outputs[1].device().is_cuda());
  REQUIRE(outputs[1].device() == torch::Device(torch::kCUDA, 0));

  REQUIRE(outputs[2].device().is_cpu());
}

TEST_CASE(
    "Parallel/DataParallelPlacesTheOutputOnTheRequestedDevice",
    "[cuda]") {
  Linear linear(3, 4);
  auto input = torch::ones({10, 3});
  auto output = parallel::data_parallel(
      linear,
      input,
      /*devices=*/at::nullopt,
      /*output_device=*/torch::Device(torch::kCUDA, 1));
  REQUIRE(output.defined());
  REQUIRE(output.device().is_cuda());
  REQUIRE(output.device().index() == 1);
}

TEST_CASE("Parallel/DataParallelUsesAllAvailableCUDADevices", "[cuda]") {
  struct M : torch::nn::Cloneable<M> {
    void reset() override {}
    torch::Tensor forward(torch::Tensor input) {
      devices->push_back(torch::DefaultTensorOptions::get().device());
      return input;
    }
    std::shared_ptr<std::vector<torch::Device>> devices;
  };

  auto m = std::make_shared<M>();
  m->devices = std::make_shared<std::vector<torch::Device>>();

  auto input = torch::ones({10, 3});
  auto output = parallel::data_parallel(m, input);

  const auto device_count = torch::cuda::device_count();
  REQUIRE(m->devices->size() == device_count);
  for (size_t i = 0; i < device_count; ++i) {
    REQUIRE(m->devices->at(i).index() == i);
  }
}

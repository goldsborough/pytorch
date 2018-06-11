#include "catch.hpp"

#include <ATen/Context.h>
#include <ATen/Functions.h>
#include <ATen/TensorOptions.h>

using namespace at;

// A macro so we don't lose location information when an assertion fails.
#define REQUIRE_OPTIONS(device_, index_, type_, layout_)                    \
  REQUIRE(options.device().type() == Device((device_), (index_)).type());   \
  REQUIRE(options.device().index() == Device((device_), (index_)).index()); \
  REQUIRE(options.dtype() == (type_));                                      \
  REQUIRE(options.layout() == (layout_))

TEST_CASE("TensorOptions/DefaultsToTheRightValues") {
  TensorOptions options;
  REQUIRE_OPTIONS(kCPU, nullopt, kFloat, kStrided);
}

TEST_CASE("TensorOptions/ReturnsTheCorrectType") {
  auto options = TensorOptions().device(kCPU).dtype(kInt).layout(kSparse);
  REQUIRE(options.type() == getType(kSparseCPU, kInt));
}

TEST_CASE("TensorOptions/UtilityFunctionsReturnTheRightTensorOptions") {
  auto options = dtype(kInt);
  REQUIRE_OPTIONS(kCPU, nullopt, kInt, kStrided);

  options = layout(kSparse);
  REQUIRE_OPTIONS(kCPU, nullopt, kFloat, kSparse);

  options = device({kCUDA, 1});
  REQUIRE_OPTIONS(kCUDA, 1, kFloat, kStrided);

  options = device_index(1);
  REQUIRE_OPTIONS(kCUDA, 1, kFloat, kStrided);

  options = dtype(kByte).layout(kSparse).device({kCUDA, 2}).device_index(3);
  REQUIRE_OPTIONS(kCUDA, 3, kByte, kSparse);
}

TEST_CASE("TensorOptions/ConstructsWellFromCPUTypes") {
  auto options = TensorOptions();
  REQUIRE_OPTIONS(kCPU, nullopt, kFloat, kStrided);

  options = TensorOptions(kInt);
  REQUIRE_OPTIONS(kCPU, nullopt, kInt, kStrided);

  options = TensorOptions(getType(kSparseCPU, kFloat));
  REQUIRE_OPTIONS(kCPU, nullopt, kFloat, kSparse);

  options = TensorOptions(getType(kSparseCPU, kByte));
  REQUIRE_OPTIONS(kCPU, nullopt, kByte, kSparse);
}

TEST_CASE("TensorOptions/ConstructsWellFromCPUTensors") {
  auto options = TensorOptions(empty(5, kDouble));
  REQUIRE_OPTIONS(kCPU, nullopt, kDouble, kStrided);

  options = TensorOptions(empty(5, getType(kSparseCPU, kByte)));
  REQUIRE_OPTIONS(kCPU, nullopt, kByte, kSparse);
}

TEST_CASE("Device/ParsesCorrectlyFromString") {
  Device device("cpu:0");
  REQUIRE(device == Device(kCPU, 0));

  device = Device("cpu");
  REQUIRE(device == Device(kCPU));

  device = Device("cuda:123");
  REQUIRE(device == Device(kCUDA, 123));

  device = Device("cuda");
  REQUIRE(device == Device(kCUDA));

  device = Device("3");
  REQUIRE(device == Device(kCUDA, 3));

  device = Device("");
  REQUIRE(device == Device(kCPU));

  std::vector<std::string> badnesses = {
      "cud:1", "cuda:", "cpu::1", ":1", "tpu:4", "??"};
  for (const auto& badness : badnesses) {
    REQUIRE_THROWS(Device(badness));
  }
}

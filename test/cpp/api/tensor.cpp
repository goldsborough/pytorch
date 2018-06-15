#include <catch.hpp>

#include <ATen/ATen.h>

using namespace at;

// A macro so we don't lose location information when an assertion fails.
#define REQUIRE_TENSOR_OPTIONS(device_, index_, type_, layout_)             \
  REQUIRE(Device(tensor).type() == Device((device_), (index_)).type());   \
  REQUIRE(Device(tensor).index() == Device((device_), (index_)).index()); \
  REQUIRE(tensor.type().scalarType() == (type_));                                      \
  REQUIRE(tensor.type().layout() == (layout_))

TEST_CASE("DefaultTensorOptions/OptionsGuard") {
  Tensor tensor;
  {
    OptionsGuard guard(TensorOptions());
    tensor = at::empty({10});
  }
  REQUIRE_TENSOR_OPTIONS(kCPU, -1, kFloat, kStrided);

  {
    OptionsGuard guard(TensorOptions().dtype(kInt));
    tensor = at::empty({10});
  }
  REQUIRE_TENSOR_OPTIONS(kCPU, -1, kInt, kStrided);

  {
    OptionsGuard guard(TensorOptions().dtype(kInt).layout(kSparse));
    tensor = at::empty({10});
  }
  REQUIRE_TENSOR_OPTIONS(kCPU, -1, kInt, kSparse);

  {
    OptionsGuard guard(dtype(kInt));
    tensor = at::empty({10});
  }
  REQUIRE_TENSOR_OPTIONS(kCPU, -1, kInt, kStrided);
}

TEST_CASE("DefaultTensorOptions/OptionsGuardCUDA", "[cuda]") {
  Tensor tensor;
  {
    OptionsGuard guard(device(kCUDA));
    tensor = at::empty({10});
  }
  REQUIRE_TENSOR_OPTIONS(kCUDA, 0, kFloat, kStrided);

  {
    OptionsGuard guard(device({kCUDA, 1}));
    tensor = at::empty({10});
  }
  REQUIRE_TENSOR_OPTIONS(kCUDA, 1, kFloat, kStrided);

  {
    OptionsGuard guard(device(kCUDA).dtype(kInt));
    tensor = at::empty({10});
  }
  REQUIRE_TENSOR_OPTIONS(kCUDA, 0, kInt, kStrided);
}

TEST_CASE("DefaultTensorOptions/DeviceGuardOptionsGuardInteraction", "[cuda]") {
  Tensor tensor;
  {
    // Check that OptionsGuard respects any active device before construction.
    DeviceGuard guard(1);
    {
      OptionsGuard guard(device(kCUDA));
      tensor = at::empty({10});
      REQUIRE_TENSOR_OPTIONS(kCUDA, 1, kFloat, kStrided);
      {
        // Check that OptionsGuard respects any active device after
        // construction.
        DeviceGuard guard(0);
        tensor = at::empty({10});
        REQUIRE_TENSOR_OPTIONS(kCUDA, 0, kFloat, kStrided);
      }
    }
  }
}

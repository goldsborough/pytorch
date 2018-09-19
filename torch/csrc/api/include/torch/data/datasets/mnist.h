#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/tensor.h>

#include <ATen/Error.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

namespace torch {
namespace data {
namespace datasets {
namespace detail {
static constexpr uint32_t kTrainSize = 60000;
static constexpr uint32_t kTestSize = 10000;
static constexpr uint32_t kImageMagicNumber = 2051;
static constexpr uint32_t kLabelMagicNumber = 2049;
static constexpr uint32_t kImageRows = 28;
static constexpr uint32_t kImageColumns = 28;
static constexpr const char* kTrainImagesFilename = "train-images-idx3-ubyte";
static constexpr const char* kTrainLabelsFilename = "train-labels-idx1-ubyte";
static constexpr const char* kTestImagesFilename = "t10k-images-idx3-ubyte";
static constexpr const char* kTestLabelsFilename = "t10k-labels-idx1-ubyte";

bool check_is_little_endian() {
  const uint32_t word = 1;
  return reinterpret_cast<const uint8_t*>(&word)[0] == 1;
}

constexpr uint32_t flip_endianness(uint32_t value) {
  return ((value & 0xff) << 24) | ((value & 0xff00) << 8) |
      ((value & 0xff0000) >> 8) | ((value & 0xff000000) >> 24);
}

uint32_t read_int32(std::ifstream& stream) {
  static const bool is_little_endian = check_is_little_endian();
  uint32_t value;
  AT_ASSERT(stream.read(reinterpret_cast<char*>(&value), sizeof value));
  return is_little_endian ? flip_endianness(value) : value;
}

uint32_t expect_int32(std::ifstream& stream, uint32_t expected) {
  const auto value = read_int32(stream);
  // clang-format off
  AT_CHECK(value == expected,
      "Expected to read number ", expected, " but found ", value, " instead");
  // clang-format on
  return value;
}

std::string join_paths(std::string head, std::string tail) {
  if (head.back() != '/') {
    head.push_back('/');
  }
  head += std::move(tail);
  return head;
}

Tensor read_images(const std::string& root, bool train) {
  const auto path =
      join_paths(root, train ? kTrainImagesFilename : kTestImagesFilename);
  std::ifstream images(path, std::ios::binary);
  AT_CHECK(images, "Error opening images file at ", path);

  const auto count = train ? kTrainSize : kTestSize;

  // From http://yann.lecun.com/exdb/mnist/
  expect_int32(images, kImageMagicNumber);
  expect_int32(images, count);
  expect_int32(images, kImageRows);
  expect_int32(images, kImageColumns);

  std::vector<char> buffer(count * kImageRows * kImageColumns);
  images.read(buffer.data(), buffer.size());

  return torch::from_blob(buffer.data(), buffer.size(), torch::kByte)
      .reshape({count, kImageRows, kImageColumns})
      .to(torch::kFloat32)
      .div(255);
}

Tensor read_labels(const std::string& root, bool train) {
  const auto path =
      join_paths(root, train ? kTrainLabelsFilename : kTestLabelsFilename);
  std::ifstream labels(path, std::ios::binary);
  AT_CHECK(labels, "Error opening labels file at ", path);

  const auto count = train ? kTrainSize : kTestSize;

  expect_int32(labels, kLabelMagicNumber);
  expect_int32(labels, count);

  std::vector<char> buffer(count);
  labels.read(buffer.data(), count);
  return torch::from_blob(buffer.data(), count, torch::kByte).to(torch::kInt64);
}
} // namespace detail

class MNIST : public Dataset<MNIST> {
 public:
  explicit MNIST(const std::string& root, bool train = true)
      : images_(detail::read_images(root, train)),
        labels_(detail::read_labels(root, train)) {}

  Example<> index(size_t index) override {
    return {images_[index], labels_[index]};
  }

  size_t size() const override {
    return images_.size(0);
  }

  bool is_train() const noexcept {
    return size() == detail::kTrainSize;
  }

 private:
  Tensor images_;
  Tensor labels_;
};
} // namespace datasets
} // namespace data
} // namespace torch

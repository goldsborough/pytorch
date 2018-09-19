#pragma once

#include <torch/data/datasets/base.h>

#include <ATen/core/ArrayRef.h>

#include <cstddef>
#include <utility>

namespace torch {
namespace data {
namespace datasets {

template <typename S, typename T>
struct Map : BatchDataset<Map<S, T>, typename T::OutputBatchType> {
  using DatasetType = S;
  using TransformType = T;

  Map(DatasetType dataset, TransformType transform)
      : dataset(std::move(dataset)), transform(std::move(transform)) {}

  typename T::OutputBatchType batch(at::ArrayRef<size_t> indices) override {
    return transform.apply_batch(dataset.batch(indices));
  }

  size_t size() const override {
    return dataset.size();
  }

  S dataset;
  T transform;
};

template <typename DatasetType, typename TransformType>
Map<DatasetType, TransformType> map(
    DatasetType&& dataset,
    TransformType&& transform) {
  static_assert(
      std::is_same<
          typename DatasetType::BatchType,
          typename TransformType::InputBatchType>::value,
      "BatchType type of dataset does not match input type of transform");
  return {std::forward<DatasetType>(dataset),
          std::forward<TransformType>(transform)};
}

} // namespace datasets
} // namespace data
} // namespace torch

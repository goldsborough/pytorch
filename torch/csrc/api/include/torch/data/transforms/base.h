#pragma once

#include <utility>
#include <vector>

namespace torch {
namespace data {
namespace transforms {
template <typename IB, typename OB>
class BatchTransform {
 public:
  using InputBatchType = IB;
  using OutputBatchType = OB;

  virtual ~BatchTransform() = default;
  virtual OutputBatchType apply_batch(InputBatchType input_batch) = 0;
};

template <
    typename I,
    typename O,
    typename IB = std::vector<I>,
    typename OB = std::vector<O>>
class Transform : public BatchTransform<IB, OB> {
 public:
  using InputType = I;
  using OutputType = O;
  using typename BatchTransform<IB, OB>::InputBatchType;
  using typename BatchTransform<IB, OB>::OutputBatchType;

  virtual OutputType apply(InputType input) = 0;

  // OutputBatchType apply_batch(InputBatchType input_batch) override {
  //   OutputBatchType output_batch;
  //   torch::detail::reserve_capacity(output_batch, input_batch.size());
  //   for (auto&& input : input_batch) {
  //     output_batch.insert(output_batch.end(), apply(std::move(input)));
  //   }
  //   return output_batch;
  // }
};
} // namespace transforms
} // namespace data
} // namespace torch

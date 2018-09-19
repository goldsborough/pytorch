#pragma once

#include <ATen/optional.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace data {
namespace detail {
namespace sequencers {
template <typename R>
struct Sequencer {
  using ResultType = R;
  using ResultProducer = std::function<at::optional<ResultType>()>;
  virtual ~Sequencer() = default;
  virtual at::optional<ResultType> next(ResultProducer next_result) = 0;
};

template <typename R>
struct NoSequencer : public Sequencer<R> {
  using typename Sequencer<R>::ResultType;
  using typename Sequencer<R>::ResultProducer;
  at::optional<ResultType> next(ResultProducer next_result) override {
    return next_result();
  }
};

template <typename R>
struct OrderedSequencer : public Sequencer<R> {
  using typename Sequencer<R>::ResultType;
  using typename Sequencer<R>::ResultProducer;

  explicit OrderedSequencer(size_t max_jobs) : buffer_(max_jobs) {}

  at::optional<ResultType> next(ResultProducer next_result) override {
    while (true) {
      if (auto& maybe_result = buffer(next_sequence_number_)) {
        auto result = std::move(*maybe_result);
        buffer(next_sequence_number_++) = at::nullopt;
        return result;
      }
      if (auto result = next_result()) {
        if (result->sequence_number == next_sequence_number_) {
          ++next_sequence_number_;
          return result;
        }
        // Stash the result for later.
        buffer(result->sequence_number) = std::move(result);
      } else {
        return at::nullopt;
      }
    }
  }

  at::optional<R>& buffer(size_t index) {
    return buffer_[index % buffer_.size()];
  }

  size_t next_sequence_number_{0};
  std::vector<at::optional<R>> buffer_;
};
} // namespace sequencers
} // namespace detail
} // namespace data
} // namespace torch

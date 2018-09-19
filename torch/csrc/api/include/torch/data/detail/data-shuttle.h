#pragma once

#include <torch/data/detail/queue.h>

#include <ATen/Error.h>
#include <ATen/optional.h>

#include <chrono>
#include <queue>
#include <utility>

namespace torch {
namespace data {
namespace detail {

template <typename Job, typename Result>
class DataShuttle {
 public:
  using JobType = Job;
  using ResultType = Result;

  enum class Event { kPush, kExhausted };

  void push_job(Job&& job) {
    jobs_.push(std::move(job));
    log_.push(Event::kPush);
  }

  void push_result(Result&& result) {
    results_.push(std::move(result));
  }

  Job pop_job() {
    return jobs_.pop();
  }

  at::optional<Result> pop_result(
      at::optional<std::chrono::seconds> timeout = at::nullopt) {
    const auto event = log_.front();
    log_.pop();
    switch (event) {
      case Event::kPush:
        return results_.pop(timeout);
      case Event::kExhausted:
        return at::nullopt;
      default:
        AT_ERROR("Unhandled event");
    }
  }

  void exhausted() {
    log_.push(Event::kExhausted);
  }

  void drain() {
    // Clear all inputs so that no further jobs are scheduled.
    auto number_cleared = jobs_.clear();
    // For each input we cleared, we will have one less pop.
    while (number_cleared-- > 0) {
      AT_ASSERT(!log_.empty() && log_.front() == Event::kPush);
      log_.pop();
    }
    // Remove any outstanding results.
    while (!log_.empty()) {
      pop_result();
    }
  }

 private:
  Queue<Job> jobs_;
  std::queue<Event> log_;
  Queue<Result> results_;
};

} // namespace detail
} // namespace data
} // namespace torch

#pragma once

#include <ATen/Error.h>
#include <ATen/optional.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>

namespace torch {
namespace data {
namespace detail {
template <typename T>
class Queue {
 public:
  void push(T&& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(std::move(value));
    cv_.notify_one();
  }

  T pop(at::optional<std::chrono::seconds> timeout = at::nullopt) {
    T value;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (timeout) {
        if (!cv_.wait_for(
                lock, *timeout, [this] { return !this->queue_.empty(); })) {
          AT_ERROR("Timeout while waiting for job result");
        }
      } else {
        cv_.wait(lock, [this] { return !this->queue_.empty(); });
      }
      assert(!queue_.empty());
      value = queue_.front();
      queue_.pop();
    }
    return value;
  }

  size_t clear() {
    std::lock_guard<std::mutex> lock(this->mutex_);
    const auto size = queue_.size();
    while (!queue_.empty()) {
      queue_.pop();
    }
    return size;
  }

 protected:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
};
} // namespace detail
} // namespace data
} // namespace torch

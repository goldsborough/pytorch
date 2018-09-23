#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace torch {
namespace detail {

// Reserve memory for a generic container.

template <typename Container>
void reserve_capacity(Container& container, size_t capacity) {}

template <typename T, typename A = std::allocator<T>>
void reserve_capacity(std::vector<T, A>& vector, size_t capacity) {
  vector.reserve(capacity);
}
} // namespace detail
} // namespace torch

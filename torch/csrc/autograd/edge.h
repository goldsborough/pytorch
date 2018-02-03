#pragma once

#include <cstdint>
#include <functional>
#include <memory>

#include "torch/csrc/utils/hash.h"

namespace torch {
namespace autograd {

class Function;

/// Represents a particular input of a function.
struct Edge {
  explicit Edge(
      const std::shared_ptr<Function>& function_ = nullptr,
      uint32_t input_nr_ = 0)
      : function(function_), input_nr(input_nr_) {}

  // Required for use in associative containers.
  bool operator==(const Edge& other) const noexcept {
    return this->function == other.function && this->input_nr == other.input_nr;
  }

  bool operator!=(const Edge& other) const noexcept {
    return !(*this == other);
  }

  /// The function this `Edge` points to.
  std::shared_ptr<Function> function;

  /// The identifier of a particular input to the function.
  uint32_t input_nr;
};
} // namespace autograd
} // namespace torch

// The idiomatic way of enabling use of a custom type as the key of hash
// containers in C++11. This method removes the requirement of having to pass
// a custom hasher to std::unordered_{map, set}.
// See http://en.cppreference.com/w/cpp/utility/hash for more information.
namespace std {
template <>
struct hash<torch::autograd::Edge> {
  // These type aliases are required by the standard.
  using argument_type = torch::autograd::Edge;
  using return_type = size_t;
  return_type operator()(const argument_type& edge) const noexcept {
    return torch::get_hash(edge.function, edge.input_nr);
  }
};
} // namespace std

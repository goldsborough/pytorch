#include "torch/csrc/autograd/saved_variable.h"

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/jit/tracer_state.h"

#include <ATen/Tensor.h>

#include <cstdint>
#include <list>
#include <memory>

namespace torch {
namespace autograd {

SavedVariable::SavedVariable(const Variable& variable, bool is_output) {
  if (!variable.defined()) {
    return;
  }
  was_default_constructed_ = false;

  data_ = variable.data();
  requires_grad_ = variable.requires_grad();
  version_counter_ = variable.version_counter();
  saved_version_ = version_counter_.current_version();
  has_grad_fn_ = !variable.is_leaf();
  output_nr_ = variable.output_nr();
  if (!has_grad_fn_) {
    grad_accumulator_ = variable.grad_accumulator();
  }
  if (!is_output) {
    grad_fn_ = variable.grad_fn();
  }
  if (variable.has_tracing_state()) {
    tracing_state_.reset(
        new jit::tracer::ValueTracingState(variable.tracing_state()));
  }
}

SavedVariable::~SavedVariable() = default;

Variable SavedVariable::unpack(std::shared_ptr<Function> saved_for) const {
  if (!data_.defined()) {
    if (!was_default_constructed_) {
      throw std::runtime_error(ERR_BACKWARD_TWICE);
    }
    return Variable();
  }

  if (saved_version_ != version_counter_.current_version()) {
    throw std::runtime_error(
        "one of the variables needed for gradient computation has been "
        "modified by an inplace operation");
  }

  auto grad_fn = grad_fn_;
  if (has_grad_fn_ && !grad_fn) {
    if (!saved_for) {
      // If saving the grad_fn would create a circular reference, then it must
      // be passed in to the unpack function.
      throw std::runtime_error("No grad_fn for non-leaf saved variable");
    }
    grad_fn = std::move(saved_for);
  }

  // NB: saved views are unpacked as normal Variables (not views) even though
  // they still share the same storage. This works only because we never call
  // in-place functions on unpacked variables.
  Variable var;
  if (grad_fn) {
    var = make_variable(data_, Edge(std::move(grad_fn), output_nr_));
  } else {
    var = make_variable(data_, requires_grad_);
  }
  var.set_version(saved_version_);

  // If a Variable is a leaf (no grad_fn saved), and it requires_grad, then we
  // should have saved the grad accumulator. Even if the Variable no longer
  // alive, the accumulator should be kept alive by the references in the
  // graph).
  if (requires_grad_ && !var.grad_fn() && grad_accumulator_.expired())
    throw std::logic_error("No grad accumulator for a saved leaf!");
  var.set_grad_accumulator(grad_accumulator_);
  if (tracing_state_) {
    var.set_tracing_state(new jit::tracer::ValueTracingState(*tracing_state_));
  }

  return var;
}

const char* ERR_BACKWARD_TWICE =
    "Trying to backward through the graph a second time, but the buffers have "
    "already been freed. Specify retain_graph=True when calling backward "
    "the first time.";

} // namespace autograd
} // namespace torch

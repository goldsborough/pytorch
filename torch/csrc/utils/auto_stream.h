#pragma once

// RAII structs to set CUDA stream

#include <ATen/ATen.h>

#ifdef USE_CUDA
#include <THC/THC.h>
#endif

struct AutoStream {
#ifdef USE_CUDA
  explicit AutoStream(THCStream* stream)
      : original_stream(
            THCState_getStream(at::globalContext().lazyInitCUDA())) {
    THCStream_retain(original_stream);
    THCState_setStream(at::globalContext().lazyInitCUDA(), stream);
  }

  ~AutoStream() {
    THCState_setStream(at::globalContext().lazyInitCUDA(), original_stream);
    THCStream_free(original_stream);
  }

  THCStream* original_stream;
#endif
};

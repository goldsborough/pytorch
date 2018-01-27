#pragma once

#include "ATen/ATenGeneral.h"
#include "ATen/Half.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace at {
template <> AT_API half convert(Half aten_half);
template <> AT_API Half convert(half cuda_half);
template <> AT_API half convert(double value);
#if CUDA_VERSION >= 9000
template <> inline __half HalfFix<__half, Half>(Half h);
template <> inline Half HalfFix<Half, __half>(__half h);
#endif
} // namespace at

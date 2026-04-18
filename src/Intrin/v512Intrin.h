#pragma once
#define ALLOW_VECTOR_INTRIN_256
#include <v256Intrin.h>

#ifndef ALLOW_VECTOR_INTRIN_512
#error "Explicitly define ALLOW_VECTOR_INTRIN_128 before including this header."
#else
#if !defined(__AVX512__) && !defined(_vectorbackend_EDITOR_MODE)
#error "_vectorbackend SIMD Vector Backend requires AVX2 flag to be enabled during compilation"
#else
namespace StormST::Vectorization::Intrinsic::v512 {
	using r512i = __m512i;
	using r512f32 = __m512;
	using r512f64 = __m512d;
}
#endif
#endif

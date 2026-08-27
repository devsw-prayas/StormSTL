#pragma once
#include <StlCompiler.h>

#ifndef STL_RUNTIME_API
#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(STL_SHARED)
#define STL_RUNTIME_API __declspec(dllexport)
#else
#define STL_RUNTIME_API __declspec(dllimport)
#endif

#elif defined(__GNUC__) || defined(__clang__)
#define STL_RUNTIME_API __attribute__((visibility("default")))
#else
#define STL_RUNTIME_API
#endif
#endif

#if defined(__INTELLISENSE__) || defined(__clang_analyzer__) || defined(STL_SIMD_SCALAR)
#define STL_EDITOR_MODE 1
#else
#define STL_EDITOR_MODE 0
#endif

#if defined(__SSE__) || defined(__SSE2__) || STL_EDITOR_MODE
#define STL_SSE_SUPPORT 1

#if defined(__SSE2__) || STL_EDITOR_MODE
#define STL_SSE2_SUPPORT 1
#else
#define STL_SSE2_SUPPORT 0
#endif

#else
#define STL_SSE_SUPPORT 0
#define STL_SSE2_SUPPORT 0
#endif

#if defined(__AVX__) || defined(__AVX2__) || STL_EDITOR_MODE
#define STL_AVX_SUPPORT 2

#if defined(__AVX2__) || STL_EDITOR_MODE
#define STL_AVX2_SUPPORT 1
#else
#define STL_AVX2_SUPPORT 0
#endif

#else
#define STL_AVX_SUPPORT 0
#define STL_AVX2_SUPPORT 0
#endif

#if defined(__AVX512F__) || STL_EDITOR_MODE
#define STL_AVX512_SUPPORT 3

#if defined(__AVX512BW__) || STL_EDITOR_MODE
#define STL_AVX512BW_SUPPORT 1
#else
#define STL_AVX512BW_SUPPORT 0
#endif

#else
#define STL_AVX512_SUPPORT 0
#define STL_AVX512BW_SUPPORT 0
#endif

#ifndef STL_SSE_ACTIVE
#define STL_SSE_ACTIVE 0
#endif

#ifndef STL_AVX_ACTIVE
#define STL_AVX_ACTIVE 0
#endif

#ifndef STL_AVX512_ACTIVE
#define STL_AVX512_ACTIVE 0
#endif

// Set Preferred Backend
#define STL_PREFERRED_VEC 0

#if STL_AVX512_ACTIVE && !STL_AVX512_SUPPORT
#error "AVX512 preference but AVX512 is not available"
#else
#if STL_PREFERRED_VEC == 0
#define STL_PREFERRED_VEC STL_AVX512_SUPPORT
#endif
#endif

#if STL_AVX_ACTIVE && !STL_AVX_SUPPORT
#error "AVX preference but AVX is not available"
#else
#if STL_PREFERRED_VEC == 0
#define STL_PREFERRED_VEC STL_AVX_SUPPORT
#endif
#endif


#if STL_SSE_ACTIVE && !STL_SSE_SUPPORT
#error "SSE preference but SSE is not available"
#else
#if STL_PREFERRED_VEC == 0
#define STL_PREFERRED_VEC STL_SSE_SUPPORT
#endif
#endif

// If no preference 

#if STL_AVX512_SUPPORT && !STL_PREFERRED_VEC
#define STL_PREFERRED_VEC STL_AVX512_SUPPORT
#elif STL_AVX_SUPPORT && !STL_PREFERRED_VEC
#define STL_PREFERRED_VEC STL_AVX_SUPPORT
#elif STL_SSE_SUPPORT && !STL_PREFERRED_VEC
#define STL_PREFERRED_VEC STL_SSE_SUPPORT
#endif

// Unroll windows

#ifndef STL_SSE_UNROLL_WINDOW
#if STL_UNROLL_WINDOW == 0 
#define STL_SSE_UNROLL_WINDOW STL_DEFAULT_UNROLL_WINDOW_SSE
#else 
#define STL_SSE_UNROLL_WINDOW STL_UNROLL_WINDOW
#endif
#endif

#ifndef STL_AVX_UNROLL_WINDOW
#if STL_UNROLL_WINDOW == 0
#define STL_AVX_UNROLL_WINDOW STL_DEFAULT_UNROLL_WINDOW_AVX
#else
#define STL_AVX_UNROLL_WINDOW STL_UNROLL_WINDOW
#endif
#endif

#ifndef STL_AVX512_UNROLL_WINDOW
#if STL_UNROLL_WINDOW == 0
#define STL_AVX512_UNROLL_WINDOW STL_DEFAULT_UNROLL_WINDOW_AVX512
#else
#define STL_AVX512_UNROLL_WINDOW STL_UNROLL_WINDOW
#endif
#endif




#include <type_traits>
#include <array>

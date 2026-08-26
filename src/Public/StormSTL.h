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
#else
#define STL_SSE_SUPPORT 0
#endif

#if defined(__AVX__) || defined(__AVX2__) || STL_EDITOR_MODE
#define STL_AVX_SUPPORT 1
#else
#define STL_AVX_SUPPORT 0
#endif

#if defined(__AVX512F__) || STL_EDITOR_MODE
#define STL_AVX512_SUPPORT 1
#else
#define STL_AVX512_SUPPORT 0
#endif

#include <type_traits>
#include <array>
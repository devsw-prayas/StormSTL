#pragma once
#include <StlCompiler.h>
#if defined(STL_SHARED)

#if STL_COMPILER_MSVC
#if defined(StormSTL_EXPORTS)
#define STL_RUNTIME_API __declspec(dllexport)
#else
#define STL_RUNTIME_API __declspec(dllimport)
#endif
#elif STL_COMPILER_CLANG || STL_COMPILER_GCC
#define STL_RUNTIME_API __attribute__((visibility("default")))
#else
#define STL_RUNTIME_API
#endif

#else
// Static build -> no import/export
#define STL_RUNTIME_API
#endif

#include <type_traits>
#include <array>

#if defined(__INTELLISENSE__) || defined(__clang_analyzer__) || defined(STL_SIMD_SCALAR)
#define STL_EDITOR_MODE 1
#else
#define STL_EDITOR_MODE 0
#endif

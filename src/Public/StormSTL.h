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

#include <type_traits>
#include <array>

#if defined(__INTELLISENSE__) || defined(__clang_analyzer__) || defined(STL_SIMD_SCALAR)
#define STL_EDITOR_MODE 1
#else
#define STL_EDITOR_MODE 0
#endif

#include "StlMemoryUtils.h"
#include "StlMemory.h"

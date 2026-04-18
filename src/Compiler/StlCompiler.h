#pragma once

// Just don't mess with this header, exposes low level shit,
// Don't want system gettin' fucked when a supposedly impossible linux port occurs
// This header is from Spectra Render Engine

#if defined(__AVX512F__)
#define STL_SIMD_AVX512 1
#elif defined(__AVX2__)
#define STL_SIMD_AVX2 1
#elif defined(__SSE2__)
#define STL_SIMD_SSE2 1
#else
#define STL_SIMD_SCALAR 1
#endif

#if defined(_MSC_VER)
#define STL_COMPILER_MSVC 1
#else
#define STL_COMPILER_MSVC 0
#endif

#if defined(__clang__)
#define STL_COMPILER_CLANG 1
#else
#define STL_COMPILER_CLANG 0
#endif

#if defined(__GNUC__) && !defined(__clang__)
#define STL_COMPILER_GCC 1
#else
#define STL_COMPILER_GCC 0
#endif

#if STL_COMPILER_MSVC
#define STL_FORCEINLINE __forceinline
#define STL_NOINLINE    __declspec(noinline)
#elif STL_COMPILER_CLANG || STL_COMPILER_GCC
#define STL_FORCEINLINE inline __attribute__((always_inline))
#define STL_NOINLINE    __attribute__((noinline))
#else
#define STL_FORCEINLINE inline
#define STL_NOINLINE
#endif

#define STL_INLINE inline

#if STL_COMPILER_MSVC
#define STL_COMPILER_BARRIER() _ReadWriteBarrier()
#elif STL_COMPILER_CLANG || STL_COMPILER_GCC
#define STL_COMPILER_BARRIER() asm volatile("" ::: "memory")
#else
#define STL_COMPILER_BARRIER()
#endif

#if STL_COMPILER_MSVC
#define STL_OPTIMIZE_OFF __pragma(optimize("", off))
#define STL_OPTIMIZE_ON  __pragma(optimize("", on))
#elif STL_COMPILER_CLANG
#define STL_OPTIMIZE_OFF _Pragma("clang optimize off")
#define STL_OPTIMIZE_ON  _Pragma("clang optimize on")
#elif STL_COMPILER_GCC
#define STL_OPTIMIZE_OFF _Pragma("GCC optimize(\"O0\")")
#define STL_OPTIMIZE_ON  _Pragma("GCC optimize(\"O2\")")
#else
#define STL_OPTIMIZE_OFF
#define STL_OPTIMIZE_ON
#endif

#if STL_COMPILER_CLANG || STL_COMPILER_GCC
#define STL_LIKELY(x)   __builtin_expect(!!(x), 1)
#define STL_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define STL_LIKELY(x)   (x)
#define STL_UNLIKELY(x) (x)
#endif

#if STL_COMPILER_MSVC
#define STL_DEBUG_BREAK() __debugbreak()
#define STL_TRAP()        __debugbreak()
#elif STL_COMPILER_CLANG || STL_COMPILER_GCC
#define STL_DEBUG_BREAK() __builtin_trap()
#define STL_TRAP()        __builtin_trap()
#else
#include <cstdlib>
#define STL_DEBUG_BREAK() std::abort()
#define STL_TRAP()        std::abort()
#endif

#if STL_COMPILER_MSVC
#define STL_UNREACHABLE() __assume(0)
#elif STL_COMPILER_CLANG || STL_COMPILER_GCC
#define STL_UNREACHABLE() __builtin_unreachable()
#else
#define STL_UNREACHABLE() STL_TRAP()
#endif

#if STL_COMPILER_MSVC
#define STL_PRAGMA(x) __pragma(x)
#elif STL_COMPILER_CLANG || STL_COMPILER_GCC
#define STL_PRAGMA(x) _Pragma(#x)
#else
#define STL_PRAGMA(x)
#endif

#if STL_COMPILER_MSVC
#define STL_DIAGNOSTIC_PUSH    STL_PRAGMA(warning(push))
#define STL_DIAGNOSTIC_POP     STL_PRAGMA(warning(pop))
#define STL_DISABLE_WARNING(w) STL_PRAGMA(warning(disable : w))
#elif STL_COMPILER_CLANG
#define STL_DIAGNOSTIC_PUSH    STL_PRAGMA(clang diagnostic push)
#define STL_DIAGNOSTIC_POP     STL_PRAGMA(clang diagnostic pop)
#define STL_DISABLE_WARNING(w) STL_PRAGMA(clang diagnostic ignored w)
#elif STL_COMPILER_GCC
#define STL_DIAGNOSTIC_PUSH    STL_PRAGMA(GCC diagnostic push)
#define STL_DIAGNOSTIC_POP     STL_PRAGMA(GCC diagnostic pop)
#define STL_DISABLE_WARNING(w) STL_PRAGMA(GCC diagnostic ignored w)
#else
#define STL_DIAGNOSTIC_PUSH
#define STL_DIAGNOSTIC_POP
#define STL_DISABLE_WARNING(w)
#endif

#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(fallthrough)
#define STL_FALLTHROUGH [[fallthrough]]
#else
#define STL_FALLTHROUGH
#endif
#else
#define STL_FALLTHROUGH
#endif

#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(nodiscard)
#define STL_NODISCARD [[nodiscard]]
#if __cplusplus >= 202002L
#define STL_NODISCARD_MSG(msg) [[nodiscard(msg)]]
#else
#define STL_NODISCARD_MSG(msg) [[nodiscard]]
#endif
#else
#define STL_NODISCARD
#define STL_NODISCARD_MSG(msg)
#endif
#else
#define STL_NODISCARD
#define STL_NODISCARD_MSG(msg)
#endif

#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(maybe_unused)
#define STL_MAYBE_UNUSED [[maybe_unused]]
#else
#define STL_MAYBE_UNUSED
#endif
#else
#define STL_MAYBE_UNUSED
#endif

#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(deprecated)
#define STL_DEPRECATED [[deprecated]]
#define STL_DEPRECATED_MSG(msg) [[deprecated(msg)]]
#else
#define STL_DEPRECATED
#define STL_DEPRECATED_MSG(msg)
#endif
#else
#define STL_DEPRECATED
#define STL_DEPRECATED_MSG(msg)
#endif

#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(noreturn)
#define STL_NORETURN [[noreturn]]
#else
#define STL_NORETURN
#endif
#else
#define STL_NORETURN
#endif

#if STL_COMPILER_MSVC
#define STL_RESTRICT __restrict
#elif STL_COMPILER_CLANG || STL_COMPILER_GCC
#define STL_RESTRICT __restrict__
#else
#define STL_RESTRICT
#endif

#define STL_ALIGNAS(n) alignas(n)

#if STL_COMPILER_CLANG || STL_COMPILER_GCC
#define STL_ASSUME_ALIGNED(ptr, n) __builtin_assume_aligned((ptr), (n))
#else
#define STL_ASSUME_ALIGNED(ptr, n) __assume(reinterpret_cast<uintptr_t>(ptr) % n == 0)
#endif

#if STL_COMPILER_CLANG || STL_COMPILER_GCC
#define STL_HOT  __attribute__((hot))
#define STL_COLD __attribute__((cold))
#else
#define STL_HOT
#define STL_COLD
#endif

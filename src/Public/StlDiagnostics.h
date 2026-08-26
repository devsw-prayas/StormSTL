#pragma once
#include <StlCompiler.h>

#if defined(_DEBUG) || defined(DEBUG)
#define STL_BUILD_DEBUG 1
#define STL_BUILD_RELEASE 0
#else
#define STL_BUILD_DEBUG 0
#define STL_BUILD_RELEASE 1
#endif

#if STL_BUILD_DEBUG
#define STL_ASSERT(expr)                        \
        do {                                            \
            if (!(expr)) {                             \
                STL_DEBUG_BREAK();                 \
                STL_TRAP();                        \
            }                                           \
        } while (0)
#else
#define STL_ASSERT(expr) do { (void)sizeof(expr); } while (0)
#endif

#if STL_BUILD_DEBUG
#define STL_ASSUME(expr) STL_ASSERT(expr)
#else
#if STL_COMPILER_MSVC
#define STL_ASSUME(expr) __assume(expr)
#else
#define STL_ASSUME(expr) do { if (!(expr)) STL_UNREACHABLE(); } while (0)
#endif
#endif

#if STL_BUILD_DEBUG
#define STL_DEBUG_ASSERT(expr) STL_ASSERT(expr)
#define STL_DEBUG_ASSUME(expr) STL_ASSUME(expr)
#else
#define STL_DEBUG_ASSERT(expr) do {} while (0)
#define STL_DEBUG_ASSUME(expr) do {} while (0)
#endif

#define STL_STATIC_ASSERT(expr, msg) static_assert(expr, msg)
#define STL_UNUSED(x) (void)(x)

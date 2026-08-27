cmake_minimum_required(VERSION 3.20)
project(StormSTL LANGUAGES CXX)

include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE StormSTL_HEADERS  CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/Public/*.h")
file(GLOB_RECURSE StormSTL_SOURCE   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/Private/*.cpp")
file(GLOB_RECURSE StormSTL_INTERNAL CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/Internal/*.h")

add_library(StormSTL SHARED
    ${StormSTL_HEADERS}
    ${StormSTL_INTERNAL}
    ${StormSTL_SOURCE}
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES
    ${StormSTL_HEADERS}
    ${StormSTL_INTERNAL}
    ${StormSTL_SOURCE}
)

target_compile_definitions(StormSTL PRIVATE STL_SHARED)

target_include_directories(StormSTL PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Public
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Internal
)

target_precompile_headers(StormSTL PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/Public/StormSTL.h)

target_compile_options(StormSTL PRIVATE
)

if(MSVC)
    set(STL_AVX_FLAG "/arch:AVX")
    set(STL_AVX512_FLAG "/arch:AVX512")
else()
    set(STL_SSE_FLAG "-msse2")
    set(STL_AVX_FLAG "-mavx")
    set(STL_AVX512_FLAG "-mavx512f")
endif()

if(DEFINED STL_SSE_FLAG)
    check_cxx_compiler_flag("${STL_SSE_FLAG}" STL_COMPILER_HAS_SSE)
else()
    # SSE2 is the x64 baseline for MSVC targets.
    set(STL_COMPILER_HAS_SSE TRUE)
endif()

check_cxx_compiler_flag("${STL_AVX_FLAG}" STL_COMPILER_HAS_AVX)
check_cxx_compiler_flag("${STL_AVX512_FLAG}" STL_COMPILER_HAS_AVX512)

option(STL_PREFER_SSE "Prefer SSE when operations are vectorized" ON)

if(STL_PREFER_SSE)
    if(NOT STL_COMPILER_HAS_SSE)
        message(FATAL_ERROR "SSE preference requested, but the compiler does not support ${STL_SSE_FLAG}")
    endif()
    if(DEFINED STL_SSE_FLAG)
        target_compile_options(StormSTL PRIVATE "${STL_SSE_FLAG}")
    endif()
    target_compile_definitions(StormSTL PUBLIC STL_SSE_ACTIVE=1)
endif()

option(STL_PREFER_AVX "Prefer AVX when operations are vectorized" ON)

if(STL_PREFER_AVX)
    if(NOT STL_COMPILER_HAS_AVX)
        message(FATAL_ERROR "AVX preference requested, but the compiler does not support ${STL_AVX_FLAG}")
    endif()
    target_compile_options(StormSTL PRIVATE "${STL_AVX_FLAG}")
    target_compile_definitions(StormSTL PUBLIC STL_AVX_ACTIVE=1)
endif()

option(STL_PREFER_AVX512 "Prefer AVX512f when operations are vectorized" ON)

if(STL_PREFER_AVX512)
    if(NOT STL_COMPILER_HAS_AVX512)
        message(FATAL_ERROR "AVX-512 preference requested, but the compiler does not support ${STL_AVX512_FLAG}")
    endif()
    target_compile_options(StormSTL PRIVATE "${STL_AVX512_FLAG}")
    target_compile_definitions(StormSTL PUBLIC STL_AVX512_ACTIVE=1)
endif()

set(STL_DEFAULT_UNROLL_WINDOW_SSE 16 CACHE STRING
    "Default SSE vector instruction count per unrolled window")
set(STL_DEFAULT_UNROLL_WINDOW_AVX 8 CACHE STRING
    "Default AVX vector instruction count per unrolled window")
set(STL_DEFAULT_UNROLL_WINDOW_AVX512 4 CACHE STRING
    "Default AVX-512 vector instruction count per unrolled window")

if(STL_PREFER_AVX512)
    set(STL_DEFAULT_UNROLL_WINDOW
        "${STL_DEFAULT_UNROLL_WINDOW_AVX512}")
elseif(STL_PREFER_AVX)
    set(STL_DEFAULT_UNROLL_WINDOW
        "${STL_DEFAULT_UNROLL_WINDOW_AVX}")
else()
    set(STL_DEFAULT_UNROLL_WINDOW
        "${STL_DEFAULT_UNROLL_WINDOW_SSE}")
endif()

set(STL_UNROLL_WINDOW 0 CACHE STRING
    "Global unroll-window override; 0 uses the backend-specific default")

foreach(window
        STL_DEFAULT_UNROLL_WINDOW_SSE
        STL_DEFAULT_UNROLL_WINDOW_AVX
        STL_DEFAULT_UNROLL_WINDOW_AVX512
        STL_DEFAULT_UNROLL_WINDOW)
    if(NOT ${${window}} MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "${window} must be a positive integer")
    endif()
endforeach()

if(NOT ${STL_UNROLL_WINDOW} MATCHES "^[0-9][0-9]*$")
    message(FATAL_ERROR "STL_UNROLL_WINDOW must be a non-negative integer")
endif()

target_compile_definitions(StormSTL PUBLIC
    STL_DEFAULT_UNROLL_WINDOW_SSE=${STL_DEFAULT_UNROLL_WINDOW_SSE}
    STL_DEFAULT_UNROLL_WINDOW_AVX=${STL_DEFAULT_UNROLL_WINDOW_AVX}
    STL_DEFAULT_UNROLL_WINDOW_AVX512=${STL_DEFAULT_UNROLL_WINDOW_AVX512}
    STL_UNROLL_WINDOW=${STL_UNROLL_WINDOW}
)

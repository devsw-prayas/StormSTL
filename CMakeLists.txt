cmake_minimum_required(VERSION 3.20)
project(StormSTL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE StormSTL_HEADERS  CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/Public/*.h")
file(GLOB_RECURSE StormSTL_COMPILER  CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/Compiler/*.h")
file(GLOB_RECURSE StormSTL_INTRIN  CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/Intrin/*.h")
file(GLOB_RECURSE StormSTL_SOURCE   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/Private/*.cpp")
file(GLOB_RECURSE StormSTL_INL      CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/Public/*.inl")

add_library(StormSTL SHARED
    ${StormSTL_HEADERS}
    ${StormSTL_COMPILER}
    ${StormSTL_INTRIN}
    ${StormSTL_SOURCE}
    ${StormSTL_INL}
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES
    ${StormSTL_HEADERS}
    ${StormSTL_COMPILER}
    ${StormSTL_INTRIN}
    ${StormSTL_SOURCE}
    ${StormSTL_INL}
)

target_compile_definitions(StormSTL PRIVATE StormSTL_EXPORTS STL_SHARED)

target_include_directories(StormSTL PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Public
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Compiler
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Intrin
)

target_precompile_headers(StormSTL PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/Public/StormSTL.h)

target_compile_options(StormSTL PRIVATE
)

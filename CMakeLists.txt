cmake_minimum_required(VERSION 3.20)
project(StormSTL LANGUAGES CXX)

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

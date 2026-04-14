cmake_minimum_required(VERSION 3.20)
project(StormSTL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE StormSTL_HEADERS  CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/Public/*.h")

add_library(StormSTL INTERFACE
    ${StormSTL_HEADERS}
)

target_include_directories(StormSTL INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Public
)

if (MSVC)
    target_compile_options(StormSTL INTERFACE /W4 /permissive-)
else()
    target_compile_options(StormSTL INTERFACE -Wall -Wextra -Wpedantic)
endif()

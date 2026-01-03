cmake_minimum_required(VERSION 3.20)
project(StormSTL)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE STL_PUBLIC "src/Public/*.h")
file(GLOB_RECURSE STL_PRIVATE "src/Private/*.cpp")

add_library(StormSTL SHARED
    ${STL_PUBLIC} ${STL_PRIVATE}
  )

target_include_directories(StormSTL PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Public
)

set_target_properties(StormSTL PROPERTIES FOLDER "Base")


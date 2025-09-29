cmake_minimum_required(VERSION 3.20)
project(devswSTL)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(devswSTL SHARED
    src/Private/devswSTL.cpp
    src/Public/devswSTL.h
  )

target_include_directories(devswSTL PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Public
)

set_target_properties(devswSTL PROPERTIES FOLDER "Base")


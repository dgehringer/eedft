cmake_minimum_required(VERSION 3.29)

project(eedft)

set(CMAKE_CXX_STANDARD 23)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

if (BUILD_TESTS)
    message(STATUS "BUILD_TESTS=${BUILD_TESTS}")
    add_subdirectory(tests)
endif ()


add_executable(main
        "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
        "include/eedft/stencil/data.hpp"
)

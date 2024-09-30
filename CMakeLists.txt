cmake_minimum_required(VERSION 3.29)

project(eedft)

set(CMAKE_CXX_STANDARD 23)

set(EEDFT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(EEDFT_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")



if (BUILD_TESTS)
    message(STATUS "BUILD_TESTS=${BUILD_TESTS}")
    add_subdirectory(tests)
endif ()

include_directories("${EEDFT_INCLUDE_DIR}")
add_executable(main
        "${EEDFT_SRC_DIR}/main.cpp"
        "${EEDFT_INCLUDE_DIR}/eedft/core/stencil/data.hpp"
        "${EEDFT_INCLUDE_DIR}/eedft/core/stencil/helpers.hpp"
        "${EEDFT_INCLUDE_DIR}/eedft/core/helpers.hpp"
        "${EEDFT_INCLUDE_DIR}/eedft/core/types.hpp"
)


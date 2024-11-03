include(FetchContent)

set(EEDFT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(EEDFT_TEST_SOURCE_DIR ${PROJECT_SOURCE_DIR}/tests)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

find_package(GTest CONFIG QUIET)
if (NOT GTest_FOUND)
    FetchContent_Declare(
            GTest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.15.2
            OVERRIDE_FIND_PACKAGE
    )
    FetchContent_MakeAvailable(GTest)
endif ()

message(STATUS "${GTest_FOUND}")
include_directories(${EEDFT_INCLUDE_DIR})


enable_testing()

add_executable(test_stencil_utils
        "${EEDFT_TEST_SOURCE_DIR}/main.cpp"
        "${EEDFT_TEST_SOURCE_DIR}/test_stencil_utils.cpp"
        "${EEDFT_INCLUDE_DIR}/eedft/core/fd/helpers.h"
)
target_link_libraries(test_stencil_utils gtest gtest_main)
add_test(NAME test_functor COMMAND test_functor)



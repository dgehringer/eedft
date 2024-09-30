include(FetchContent)

set(EEDFT_INCLUDE_DIR ${PROJECT_SOURCE_DIR})
set(EEDFT_TEST_SOURCE_DIR ${PROJECT_SOURCE_DIR}/tests)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

find_package(GTest CONFIG QUIET)
if (NOT GTest_FOUND)
    FetchContent_Declare(
            GTest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        dea0216d0c6bc5e63cf5f6c8651cd268668032ec # release-1.18.0
            OVERRIDE_FIND_PACKAGE
    )
    FetchContent_MakeAvailable(GTest)
endif ()

message(STATUS "${GTest_FOUND}")
include_directories(${EEDFT_INCLUDE_DIR})


enable_testing()

add_executable(test_functor
        "${EEDFT_TEST_SOURCE_DIR}/main.cpp"
        "${EEDFT_TEST_SOURCE_DIR}/test_functor.cpp"
        #"${EEDFT_INCLUDE_DIR}/funcutils/functor/functor.hpp"
)
target_link_libraries(test_functor gtest gtest_main)
add_test(NAME test_functor COMMAND test_functor)



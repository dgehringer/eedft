
include(FetchContent)
FetchContent_Declare(
  googletest
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/google/googletest/archive/refs/heads/v1.14.x.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


set(EEDFT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set(EEDFT_TEST_SOURCE_DIR ${PROJECT_SOURCE_DIR}/tests)
set(EEDFT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

file(GLOB sources "${EEDFT_SOURCE_DIR}/*.c")
list(REMOVE_ITEM sources "${EEDFT_SOURCE_DIR}/main.c")

file(GLOB tests "${EEDFT_TEST_SOURCE_DIR}/*.cpp")
list(REMOVE_ITEM tests "${EEDFT_TEST_SOURCE_DIR}/main.cpp")


include_directories(${EEDFT_INCLUDE_DIR})

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fno-inline-functions -fno-inline-small-functions -DEEDFT_TESTING")
SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -fno-inline-functions -fno-inline-small-functions -DEEDFT_TESTING")

foreach(file ${tests})
  set(name)
  get_filename_component(name ${file} NAME_WE)
  add_executable("${name}"
    ${sources}
    ${file}
    "${EEDFT_TEST_SOURCE_DIR}/main.cpp")
  target_link_libraries("${name}" gtest_main)
  add_test(NAME ${name} COMMAND "${name}")
endforeach()
  

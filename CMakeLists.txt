cmake_minimum_required(VERSION 3.1)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(eedft)

enable_testing()
add_subdirectory(tests)

##
### Source definitions ###
##

set(EEDFT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set(EEDFT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

include_directories(${EEDFT_INCLUDE_DIR})

add_executable(eedft
  ${EEDFT_SOURCE_DIR}/stencil.c
  ${EEDFT_SOURCE_DIR}/kernel.c
  ${EEDFT_SOURCE_DIR}/main.c
  )
target_link_libraries(eedft m)

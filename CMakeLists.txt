cmake_minimum_required(VERSION 3.5.0)

project(voxelyze.3.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_CURRENT_LIST_DIR}/src/VX3/voxelyzeManager)
include_directories(${CMAKE_CURRENT_LIST_DIR}/src/VX3/tools)
include_directories(${CMAKE_CURRENT_LIST_DIR}/src/VX3)
include_directories(${CMAKE_CURRENT_LIST_DIR}/src/VX2)
include_directories(${CMAKE_CURRENT_LIST_DIR}/src/VX1)
file( GLOB voxelyzeManager ${CMAKE_CURRENT_LIST_DIR}/src/VX3/voxelyzeManager/main.c*)
file( GLOB VX3_SRC_VOXELYZEMANAGER ${CMAKE_CURRENT_LIST_DIR}/src/VX3/voxelyzeManager/VX3_*.c* )
file( GLOB VX3_SRC_TOOLS ${CMAKE_CURRENT_LIST_DIR}/src/VX3/tools/*.c* )
file( GLOB VX3_SRC ${CMAKE_CURRENT_LIST_DIR}/src/VX3/*.c* )
file( GLOB VX2_SRC ${CMAKE_CURRENT_LIST_DIR}/src/VX2/*.c* )
file( GLOB VX1_SRC ${CMAKE_CURRENT_LIST_DIR}/src/VX1/*.c* )

add_library(voxelyze.3.0 ${VX3_SRC_VOXELYZEMANAGER} ${VX3_SRC_TOOLS} ${VX3_SRC} ${VX2_SRC} ${VX1_SRC})

add_executable(voxelyzeManager ${voxelyzeManager})
target_link_libraries(voxelyzeManager PUBLIC voxelyze.3.0)

# for Boost
set(Boost_USE_STATIC_LIBS   ON)
find_package(Boost REQUIRED COMPONENTS filesystem thread chrono)
target_link_libraries(voxelyze.3.0 PUBLIC ${Boost_LIBRARIES})
target_link_libraries(voxelyzeManager PUBLIC ${Boost_LIBRARIES})

#for zlib
add_compile_definitions(USE_ZLIB_COMPRESSION)
find_package(ZLIB)
target_link_libraries(voxelyze.3.0 PRIVATE ${ZLIB_LIBRARIES})

# CUDA options
set_target_properties(voxelyze.3.0 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(voxelyzeManager PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_compile_options(voxelyze.3.0 PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-G -arch=compute_60 -code=compute_60>)
# target_compile_options(voxelyzeManager PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-G -arch=compute_60 -code=compute_60>)
target_compile_options(voxelyze.3.0 PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: -arch=compute_60 -code=compute_60>)
target_compile_options(voxelyzeManager PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: -arch=compute_60 -code=compute_60>)
list(APPEND CMAKE_CUDA_FLAGS "-gencode=arch=compute_60,code=sm_60")


#for gtest
################################
# Testing
################################
option(test "Build all tests." ON) # Makes boolean 'test' available.
if (test)
  if (APPLE)
    add_definitions(-DGTEST_USE_OWN_TR1_TUPLE)
    add_definitions(-D__GLIBCXX__)
  endif (APPLE)

  # This adds another subdirectory, which has 'project(gtest)'.
  add_subdirectory(test/googletest)

  enable_testing()

  # Include the gtest library. gtest_SOURCE_DIR is available due to
  # 'project(gtest)' above.
  include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

  ##############
  # Unit Tests
  ##############
  set(TEST_DIR test/testCase)
  file(GLOB TEST_FILES ${TEST_DIR}/test_*.c*)
  add_executable(runUnitTests ${TEST_FILES})

  # Standard linking to gtest stuff.
  target_link_libraries(runUnitTests gtest gtest_main)

  # Extra linking for the project.
  target_link_libraries(runUnitTests voxelyze.3.0)

  # This is so you can do 'make test' to see all your tests run, instead of
  # manually running the executable runUnitTests to see those specific tests.
  add_test(NAME that-test-I-made COMMAND runUnitTests)

  # You can also omit NAME and COMMAND. The second argument could be some other
  # test executable.
  add_test(that-other-test-I-made runUnitTests)
endif()

set(CMAKE_VERBOSE_MAKEFILE OFF)
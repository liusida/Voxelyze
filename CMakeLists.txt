cmake_minimum_required(VERSION 3.5.0)

project(voxelyze.3.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_CURRENT_LIST_DIR}/src/VX3/voxelyzeManager)
include_directories(${CMAKE_CURRENT_LIST_DIR}/src/VX3)
include_directories(${CMAKE_CURRENT_LIST_DIR}/src/VX2)
include_directories(${CMAKE_CURRENT_LIST_DIR}/src/VX1)
file( GLOB voxelyzeManager ${CMAKE_CURRENT_LIST_DIR}/src/VX3/voxelyzeManager/*.c*)
file( GLOB VX3_SRC ${CMAKE_CURRENT_LIST_DIR}/src/VX3/*.c* )
file( GLOB VX2_SRC ${CMAKE_CURRENT_LIST_DIR}/src/VX2/*.c* )
file( GLOB VX1_SRC ${CMAKE_CURRENT_LIST_DIR}/src/VX1/*.c* )

add_library(voxelyze.3.0 ${SRC} ${VX3_SRC} ${VX2_SRC} ${VX1_SRC})

add_executable(voxelyzeManager ${voxelyzeManager})
target_link_libraries(voxelyzeManager PUBLIC voxelyze.3.0)

set(Boost_USE_STATIC_LIBS   ON)
find_package(Boost REQUIRED COMPONENTS filesystem thread)
target_link_libraries(voxelyzeManager PUBLIC ${Boost_LIBRARIES})

add_compile_definitions(USE_ZLIB_COMPRESSION)
#for zlib
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

set(CMAKE_VERBOSE_MAKEFILE OFF)
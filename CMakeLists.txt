cmake_minimum_required(VERSION 3.0.0)
project(Voxelyze VERSION 0.1.0)

# include(CTest)
# enable_testing()

set(INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include)
file( GLOB SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.C*)

add_executable(Voxelyze main.cpp ${SRC})

target_include_directories(Voxelyze PUBLIC ${INCLUDE_DIR})

# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)

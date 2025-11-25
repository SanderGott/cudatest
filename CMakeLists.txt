cmake_minimum_required(VERSION 3.10)
project(A3)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

enable_language(CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

cmake_policy(SET CMP0104 NEW)

# compile for all for now. probably remove later 
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 75 80 86)

find_package(SDL2 REQUIRED)
#find_package(OpenMP REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

add_executable(cpu main.cpp)
add_executable(cuda main.cpp cuda.cu)
add_executable(mandel main.cpp mandel.cpp)
add_executable(mandelcuda main.cpp mandelcuda.cu)

target_compile_definitions(cuda PRIVATE cuda)
target_compile_definitions(cpu PRIVATE cpu)
target_compile_definitions(mandel PRIVATE mandel)
target_compile_definitions(mandelcuda PRIVATE mandelcuda)



target_link_libraries(cuda ${SDL2_LIBRARIES})
target_link_libraries(cpu ${SDL2_LIBRARIES})
target_link_libraries(mandel ${SDL2_LIBRARIES})
target_link_libraries(mandelcuda ${SDL2_LIBRARIES})



#target_link_libraries(cuda OpenMP::OpenMP_CXX)

set_target_properties(cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuda PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(mandelcuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(mandelcuda PROPERTIES LINKER_LANGUAGE CUDA)
# add_executable(controller controller.cpp)
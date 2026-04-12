cmake_minimum_required(VERSION 3.22)
project(openmini-native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(OPENMINI_ENABLE_CUDA "Enable CUDA support" OFF)
option(OPENMINI_ENABLE_METAL "Enable Metal support (macOS)" ON)
option(OPENMINI_ENABLE_BITNET "Enable BitNet LUT support" ON)

if(OPENMINI_ENABLE_CUDA)
    enable_language(CUDA)
    message(STATUS "CUDA enabled")
endif()

if(OPENMINI_ENABLE_METAL AND APPLE)
    find_library(METAL_FRAMEWORK REQUIRED Metal)
    message(STATUS "Metal framework found: ${METAL_FRAMEWORK}")
endif()

find_package(Threads REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_compile_options(
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall>
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wextra>
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O2>
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Release>:-O3>
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/src
)

add_library(openmini-core STATIC
    src/engine.cpp
    src/model_loader.cpp
    src/inference.cpp
    src/kv_cache.cpp
    src/sampling.cpp
    src/openmini_api.cpp
)

target_compile_definitions(openmini-core PUBLIC
    OPENMINI_VERSION_MAJOR=2
    OPENMINI_VERSION_MINOR=0
    OPENMINI_VERSION_PATCH=0
)

if(OPENMINI_ENABLE_BITNET)
    target_compile_definitions(openmini-core PUBLIC OPENMINI_ENABLE_BITNET)
    target_sources(openmini-core PRIVATE src/quant_lut.cpp)
endif()

add_library(openmini SHARED
    src/openmini_api.cpp
)

target_link_libraries(openmini 
    openmini-core
    Threads::Threads
    dl
    m
)

if(APPLE)
    target_link_libraries(openmini ${METAL_FRAMEWORK})
endif()

if(OPENMINI_ENABLE_CUDA)
    target_link_libraries(openmini cudart cublas)
endif()

install(TARGETS openmini LIBRARY DESTINATION lib)
install(FILES include/openmini.h DESTINATION include)

message(STATUS "")
message(STATUS "OpenMini Native Build Configuration:")
message(STATUS "  Version: 2.0.0")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA: ${OPENMINI_ENABLE_CUDA}")
message(STATUS "  Metal: ${OPENMINI_ENABLE_METAL}")
message(STATUS "  BitNet: ${OPENMINI_ENABLE_BITNET}")
message(STATUS "")

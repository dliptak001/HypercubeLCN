cmake_minimum_required(VERSION 3.20)
project(HypercubeLCN VERSION 1.1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Family convention (matches HypercubeCascade): fast-math in Release,
# everywhere — the exes and the Python wheel compile the same way.
# HYPERCUBE_ARCH controls -march (same knob as python/CMakeLists.txt).
# Defaults to "native" for local dev builds; release packaging pins
# "x86-64-v2" for portable exes; "none" skips -march (ARM, MSVC).
set(HYPERCUBE_ARCH "native" CACHE STRING
    "Target architecture for -march (native, x86-64-v2, none)")

add_compile_options("$<$<CONFIG:Release>:-O3;-ffast-math>")
if(NOT HYPERCUBE_ARCH STREQUAL "none")
    if(HYPERCUBE_ARCH STREQUAL "native")
        add_compile_options(
            "$<$<CONFIG:Release>:-march=native;-mtune=native>")
    else()
        add_compile_options(
            "$<$<CONFIG:Release>:-march=${HYPERCUBE_ARCH};-mtune=generic>")
    endif()
endif()

# The SDK surface: Core + Training, nothing else. Examples and the
# Python bindings both link this.
add_library(HypercubeLCNCore STATIC
        Core.cpp
        Training.cpp)
target_include_directories(HypercubeLCNCore PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(lcn_raman
        examples/RamanBaseline/lcn_raman.cpp
        examples/RamanBaseline/BaselineExtractor.cpp
        examples/RamanBaseline/RamanDataset.cpp)
target_link_libraries(lcn_raman PRIVATE HypercubeLCNCore)
target_include_directories(lcn_raman PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/common
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/RamanBaseline)

add_executable(lcn_raman_extract
        examples/RamanBaseline/lcn_raman_extract.cpp
        examples/RamanBaseline/BaselineExtractor.cpp
        examples/RamanBaseline/RamanDataset.cpp
        examples/RamanBaseline/RamanExtract.cpp)
target_link_libraries(lcn_raman_extract PRIVATE HypercubeLCNCore)
target_include_directories(lcn_raman_extract PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/common
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/RamanBaseline)

add_executable(lcn_raman_narrow
        examples/RamanBaselineNarrowIO/lcn_raman_narrow.cpp
        examples/RamanBaseline/BaselineExtractor.cpp
        examples/RamanBaseline/RamanDataset.cpp)
target_link_libraries(lcn_raman_narrow PRIVATE HypercubeLCNCore)
target_include_directories(lcn_raman_narrow PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/common
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/RamanBaseline
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/RamanBaselineNarrowIO)

add_executable(lcn_raman_narrow_extract
        examples/RamanBaselineNarrowIO/lcn_raman_narrow_extract.cpp
        examples/RamanBaseline/BaselineExtractor.cpp
        examples/RamanBaseline/RamanDataset.cpp
        examples/RamanBaseline/RamanExtract.cpp)
target_link_libraries(lcn_raman_narrow_extract PRIVATE HypercubeLCNCore)
target_include_directories(lcn_raman_narrow_extract PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/common
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/RamanBaseline
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/RamanBaselineNarrowIO)

add_executable(lcn_mnist
        examples/mnist/lcn_mnist.cpp
        examples/common/SpatialAug.cpp
        examples/common/SpatialEmbed.cpp)
target_link_libraries(lcn_mnist PRIVATE HypercubeLCNCore)
target_include_directories(lcn_mnist PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/common
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/mnist)

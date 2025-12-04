cmake_minimum_required(VERSION 3.18)

project(raylib_cuda_lib LANGUAGES C CXX CUDA)

# CMake policy compatibility for older raylib CMake
if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

# CUDA architecture Settings
# If the user doesn't specify the GPU architecture,
# use 'native' for current machine's GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# If RAYLIB_DIR is given: use that source tree
# Otherwise: Automatically download and build raylib
set (RAYLIB_DIR "" CACHE PATH
    "Path to a local raylib source tree (with CMakeLists.txt). If empty, raylib will be fetched automatically."
)

# raylib options: Don't build examples/games to boost the speed of raylib setup
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES    OFF CACHE BOOL "" FORCE)

if(RAYLIB_DIR)
    # Use provided raylib source
    if(EXISTS "${RAYLIB_DIR}/CMakeLists.txt")
        message(STATUS "Using local raylib from: ${RAYLIB_DIR}")
        add_subdirectory("${RAYLIB_DIR}" raylib_build)
    else()
        message(FATAL ERROR 
            "RAYLIB_DIR is set to '${RAYLIB_DIR}', but no CMakeLists.txt was found in the directory."
        )
    endif()
else()
    # No path provided: fetch raylib from GitHub
    include(FetchContent)
    set(FETCHCONTENT_QUIET OFF)
    
    message(STATUS "RAYLIB_DIR not set; fetching raylib from GitHub (tag 5.0).")

    FetchContent_Declare(
        raylib
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG        5.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(raylib)
endif()

add_library(raylib_cuda STATIC
    src/raylib_cuda.c
    src/raylib_cuda_backend.cu
)

# Allow library to use raylib headers
target_link_libraries(raylib_cuda PUBLIC raylib)

# Export the 'include' folder so that raylib_cuda.h can be found by applications
target_include_directories(raylib_cuda PUBLIC include)

#Windows runtime fix
if (MSVC)
    set_property(TARGET raylib_cuda PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()


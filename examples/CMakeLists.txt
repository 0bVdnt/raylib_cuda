cmake_minimum_required(VERSION 3.18)

# Helper function to add CUDA examples
function(add_cuda_example name)
    add_executable(${name} ${name}.cu)
    target_link_libraries(${name} PRIVATE raylib_cuda)
    set_target_properties(${name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        FOLDER "Examples"
    )
endfunction()

add_cuda_example(game_of_life)

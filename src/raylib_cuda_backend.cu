/*
 * raylib_cuda_backend.cu
 * Handles windows/CUDA, no raylib allowed here to prevent conflicts.
 */

// Windows & GL Headers - No Conflicts since Raylib is not included here
#include <windows.h>
#include <gl/GL.h>

// CUDA Headers
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <string.h>

// OPTIMUS / HYBRID GPU Exports (Forcing use of CUDA Enabled GPU)
extern "C"
{
    __declspec(dllexport) unsigned long NvOptimusEnablement = 1;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

// Backend implementation
extern "C"
{
    // Return 0 on success, non-zero on failure
    int rlc_backend_check()
    {
        // Check for intel gpu, prevent crash
        const char *renderer = (const char *)glGetString(GL_RENDERER);
        if (renderer && strstr(renderer, "Intel")) {
            printf("RLC Error: Intel GPU detected (%s). Aborting.\n", renderer);
            return 2; // Error: Wrong GPU
        }

        // Check CUDA
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess || count == 0) {
            printf("RLC Error: No CUDA device found (Error %d)\n", err);
            return 1; // Error: No CUDA
        }

        cudaSetDevice(0);
        return 0; // Success
    }

    void *rlc_backend_register(unsigned int id)
    {
        cudaGraphicsResource_t res;
        cudaError_t err = cudaGraphicsGLRegisterImage(&res, id, GL_TEXTURE_2D,
                                                      cudaGraphicsRegisterFlagsSurfaceLoadStore);
        if (err != cudaSuccess) {
            printf("RLC Error: Register failed(%d)\n", err);
            return NULL;
        }
        return (void *)res;
    }

    void rlc_backend_unregister(void *res)
    {
        if (res) cudaGraphicsUnregisterResource((cudaGraphicsResource_t)res);
    }

    unsigned long long rlc_backend_map(void *res)
    {
        if (!res)
            return 0;

        cudaGraphicsResource_t g_res = (cudaGraphicsResource_t)res;

        if (cudaGraphicsMapResources(1, &g_res, 0) != cudaSuccess)
            return 0;

        cudaArray_t array;
        cudaGraphicsSubResourceGetMappedArray(&array, g_res, 0, 0);

        cudaResourceDesc desc = {};
        desc.resType = cudaResourceTypeArray;
        desc.res.array.array = array;

        cudaSurfaceObject_t surf = 0;
        cudaCreateSurfaceObject(&surf, &desc);

        return (unsigned long long)surf;
    }

    void rlc_backend_unmap(void *res, unsigned long long surf)
    {
        if (!res)
            return;
        if (surf)
            cudaDestroySurfaceObject((cudaSurfaceObject_t)surf);
        
        cudaGraphicsResource_t g_res = (cudaGraphicsResource_t)res;
        cudaGraphicsUnmapResources(1, &g_res, 0);
        cudaDeviceSynchronize();
    }
}

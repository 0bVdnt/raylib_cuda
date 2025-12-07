/*
 * raylib_cuda_backend.cu
 * CUDA/OpenGL interop backend
 * Handles platform-specific GPU operations
 */

#ifndef RLC_SKIP_GL_SYNC
#define RLC_DO_GL_SYNC 1
#else
#define RLC_DO_GL_SYNC 0
#endif

// Platform detection
#ifdef _WIN32
#define RLC_PLATFORM_WINDOWS
#include <windows.h>
#include <GL/gl.h>
#elif defined(__linux__)
#define RLC_PLATFORM_LINUX
#include <GL/gl.h>
#elif defined(__APPLE__)
#define RLC_PLATFORM_MACOS
#include <OpenGL/gl.h>
#warning "macOS support is still experimental - CUDA may not work."
#else
#error "Unsupported Platform"
#endif

// CUDA Headers
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstdio>
#include <cstring>

// ===============================================================
// GPU Selection Hints
// Forces use of discrete NVIDIA GPU on hybrid systems
// ===============================================================
#ifdef RLC_PLATFORM_WINDOWS
extern "C"
{
    __declspec(dllexport) unsigned long NvOptimusEnablement = 1;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

// ===============================================================
// Internal Logging
// ===============================================================
#define RLC_BACKEND_LOG(fmt, ...) \
    fprintf(stderr, "[RLC Backend]" fmt "\n", ##__VA_ARGS__);
#define RLC_BACKEND_ERROR(fmt, ...) \
    fprintf(stderr, "[RLC Backend ERROR]" fmt "\n", ##__VA_ARGS__);

// ===============================================================
// CUDA Error Checking helper
// ===============================================================
static bool check_cuda_error(cudaError_t err, const char *operation)
{
    if (err != cudaSuccess)
    {
        RLC_BACKEND_ERROR("%s failed: %s (code %d)",
                          operation, cudaGetErrorString(err), (int)err);
        return false;
    }
    return true;
}

// ===============================================================
// Backend implementation
// ===============================================================
extern "C"
{
    // Check for CUDA-capable GPU and initialize
    // Returns: 0 for success, 1 for No CUDA and 2 for Wrong GPU
    int rlc_backend_check(void)
    {
        // Log OpenGL renderer info
        const char *renderer = (const char *)glGetString(GL_RENDERER);
        const char *vendor = (const char *)glGetString(GL_VENDOR);

        if (renderer)
        {
            RLC_BACKEND_LOG("OpenGL Renderer: %s", renderer);
        }
        if (vendor)
        {
            RLC_BACKEND_LOG("OpenGL Vendor: %s", vendor);
        }

        // Check for Intel Integrated GPU (Common on laptops)
        // This would mean CUDA wont't work with the current GL context
        if (renderer && strstr(renderer, "Intel"))
        {
            RLC_BACKEND_ERROR("Intel GPU detected (%s)", renderer);
            RLC_BACKEND_ERROR("Please ensure your system uses the NVIDIA GPU for this application");
#ifdef RLC_PLATFORM_WINDOWS
            RLC_BACKEND_ERROR("On Windows: Right-click the exe -> Run with graphics processor -> NVIDIA");
            RLC_BACKEND_ERROR("Or set in NVIDIA Control Panel -> Manage 3D Settings -> Program Settings");
#elif defined(RLC_PLATFORM_LINUX)
            RLC_BACKEND_ERROR("On Linux: Run with __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia");
#endif
            return 2; // Error: Wrong GPU
        }

        // Check CUDA device availability
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess)
        {
            RLC_BACKEND_ERROR("cudaGetDeviceCount failed: %s", cudaGetErrorString(err));
            return 1; // Error: No CUDA
        }

        if (deviceCount == 0)
        {
            RLC_BACKEND_ERROR("No CUDA-Capable devices found");
            return 1; // Error: No CUDA
        }

        // Log available devices
        for (int i = 0; i < deviceCount; i++)
        {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            RLC_BACKEND_LOG("CUDA Device %d: %s(Compute %d.%d, %zu MB)",
                            i, prop.name, prop.major, prop.minor,
                            prop.totalGlobalMem / (1024 * 1024));
        }

        // Set device 0 as active device
        err = cudaSetDevice(0);
        if (!check_cuda_error(err, "cudaSetDevice"))
        {
            return 1;
        }

        // NOTE: cudaSetGLDevice is deprecated since CUDA 5.0
        // Modern CUDA handles GL interop context automatically
        // Just ensure we can create a test context
        err = cudaFree(0); // Lazy context initialization
        if (!check_cuda_error(err, "cudaFree(0) - context init"))
        {
            return 1;
        }

        RLC_BACKEND_LOG("CUDA backend initialized successfully.");
        return 0; // Success
    }

    // Register an OpenGL texture with CUDA
    // Returns: CUDA graphics resource handle, or NULL on failure
    void *rlc_backend_register(unsigned int textureid)
    {
        if (textureid == 0)
        {
            RLC_BACKEND_ERROR("Cannot register an invalid texture (id = 0)");
            return nullptr;
        }

        cudaGraphicsResource_t resource = nullptr;
        cudaError_t err = cudaGraphicsGLRegisterImage(&resource, textureid, GL_TEXTURE_2D,
                                                      cudaGraphicsRegisterFlagsSurfaceLoadStore);

        if (!check_cuda_error(err, "cudaGraphicsGLRegisterImage"))
        {
            return nullptr;
        }

        RLC_BACKEND_LOG("Register texture %u with CUDA", textureid);
        return static_cast<void *>(resource);
    }

    // Unregister a texture from CUDA
    void rlc_backend_unregister(void *res)
    {
        if (res == nullptr)
        {
            return;
        }
        cudaGraphicsResource_t resource = static_cast<cudaGraphicsResource_t>(res);
        cudaError_t err = cudaGraphicsUnregisterResource(resource);

        if (!check_cuda_error(err, "cudaGraphicsUnregisterResource"))
        {
            // Log but don't fail - Resource may be already freed
        }
    }

    // Map a registered resource and create a surface object for kernel access
    // Returns: Surface object handle (cudaSurfaceObject_t), or 0 on failure
    unsigned long long rlc_backend_map(void *res)
    {
        if (res == nullptr)
        {
            RLC_BACKEND_ERROR("Cannot map NULL resource");
            return 0;
        }

        cudaGraphicsResource_t resource = static_cast<cudaGraphicsResource_t>(res);
        cudaError_t err;

// Ensure OpenGL has finished all pending operations
// This prevents race conditions between GL and CUDA
#if RLC_DO_GL_SYNC
        glFinish();
#endif

        // Step 1: Map the graphics resource
        err = cudaGraphicsMapResources(1, &resource, 0);
        if (!check_cuda_error(err, "cudaGraphicsMapResources"))
        {
            return 0;
        }

        // Step 2: Get the mapped array
        cudaArray_t array = nullptr;
        err = cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0);
        if (!check_cuda_error(err, "cudaGraphicsSubResourceGetMappedArray"))
        {
            // Must unmap before returning error
            cudaGraphicsUnmapResources(1, &resource, 0);
            return 0;
        }

        // Step 3: Create surface object
        cudaResourceDesc desc = {};
        desc.resType = cudaResourceTypeArray;
        desc.res.array.array = array;

        cudaSurfaceObject_t surfObj = 0;
        err = cudaCreateSurfaceObject(&surfObj, &desc);
        if (!check_cuda_error(err, "cudaCreateSurfaceObject"))
        {
            // Must unmap before returning error
            cudaGraphicsUnmapResources(1, &resource, 0);
            return 0;
        }

        return static_cast<unsigned long long>(surfObj);
    }

    // Synchronize GPU
    void rlc_backend_sync(void)
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (!check_cuda_error(err, "cudaDeviceSynchronize"))
        {
            // Log but continue
        }
    }

    // Unmap a resource and destroy its surface object
    // sync: if true, calls cudaDeviceSynchronize after unmapping
    void rlc_backend_unmap(void *res, unsigned long long surfaceObject, bool sync)
    {
        if (res == nullptr)
        {
            return;
        }

        if (surfaceObject != 0)
        {
            cudaSurfaceObject_t surfObj = static_cast<cudaSurfaceObject_t>(surfaceObject);
            cudaError_t err = cudaDestroySurfaceObject(surfObj);
            if (!check_cuda_error(err, "cudaDestroySurfaceObject"))
            {
                // Continue anyway to unmap the resource
            }
        }

        // Unmap the graphics resource
        cudaGraphicsResource_t resource = static_cast<cudaGraphicsResource_t>(res);
        cudaError_t err = cudaGraphicsUnmapResources(1, &resource, 0);
        if (!check_cuda_error(err, "cudaGraphicsUnmapResources"))
        {
            // Log but continue
        }

        if (sync)
        {
            rlc_backend_sync();
        }
    }

    // Reset CUDA Device
    void rlc_backend_reset(void)
    {
        cudaError_t err = cudaDeviceReset();
        if (err != cudaSuccess)
        {
            RLC_BACKEND_LOG("cudaDeviceReset warning: %s", cudaGetErrorString(err));
        }
    }
} // extern "C"
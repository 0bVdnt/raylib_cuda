#ifndef RAYLIB_CUDA_H
#define RAYLIB_CUDA_H

#include "raylib.h"
#include <stdbool.h>

// To avoid name mangling while being compiled
#ifdef __cplusplus
extern "C"
{
#endif

    // =====================================================
    // 1. Versions and constants
    // =====================================================

#define RLC_VERSION_MAJOR 1
#define RLC_VERSION_MINOR 0
#define RLC_VERSION_PATCH 0

// Pixel format: RGBA8 (4 Bytes per pixel)
// When writing from CUDA kernel use:
// surf2DWrite(make_uchar4(r, g, b, a), surfObj, x * 4, y);
#define RLC_BYTES_PER_PIXEL 4

    // =====================================================
    // 2. Error Codes
    // =====================================================

    typedef enum RLC_ERROR
    {
        RLC_OK = 0,
        RLC_ERROR_NO_CUDA_DEVICE,
        RLC_ERROR_WRONG_GPU,
        RLC_ERROR_REGISTER_FAILED,
        RLC_ERROR_MAP_FAILED,
        RLC_ERROR_NOT_MAPPED,
        RLC_ERROR_ALREADY_MAPPED,
        RLC_ERROR_NULL_SURFACE
    } RLC_Error;

    // =====================================================
    // 3. Data Types
    // =====================================================

    // A Wrapper around a raylib texture + CUDA Resource
    // Note: Fields prefixed with '_' are internal - do not modify directly.
    typedef struct RLC_Surface
    {
        Texture2D texture; // Raylib texture for drawing
        int width;         // Surface width in pixels
        int height;        // Surface height in pixels

        // Internal Fields (Do not modify)
        void *_cuda_res;              // CUDA Graphics resource handle
        unsigned long long _surf_obj; // CUDA surface object (valid only when mapped)
        bool _is_mapped;
    } RLC_Surface;

    // =====================================================
    // 4. Library Management
    // =====================================================

    // Initializes CUDA and Raylib Window
    // Return true on success, false if no compatible CUDA GPU is found
    // On failure, the window is NOT left open
    bool RLC_Init(int width, int height, const char *title);

    // Closes window and cleans up all resources
    void RLC_Close();

    // Return the last error code
    RLC_Error RLC_GetLastError(void);

    // Return a human-readable error message for the given error code
    const char *RLC_ErrorString(RLC_Error error);

    // =====================================================
    // 5. Surface Management
    // =====================================================

    // Creates a surface for CUDA rendering
    // Returns a surface with _cuda_res set to NULL on failure
    // Checks with: if(surface._cuda_res == NULL) {/* handle error */}
    RLC_Surface RLC_CreateSurface(int width, int height);

    // Frees the surface and associated resources
    // Safe to call with NULL or already-freed surface
    void RLC_UnloadSurface(RLC_Surface *surface);

    // =====================================================
    // 6. Execution Pipeline
    // =====================================================

    // Begins CUDA access to the surface
    // Returns a cudaSurfaceObject_t (as unsigned long long) for use in the kernels
    // Returns 0 on failure - check RLC_GetLastError() for details
    //
    // Usage in CUDA kernel:
    // cudaSurfaceObject_t surf = (cudaSurfaceObject_t)surfObj;
    // surf2DWrite(make_uchar4(r, g, b, a), surf, x * 4, y);
    //
    // IMPORTANT: Must call RLC_EndAccess() before drawing with raylib
    unsigned long long RLC_BeginAccess(RLC_Surface *surface);

    // Ends CUDA access and Synchronizes
    // Must be called after RLC_BeginAccess() before using texture in with Raylib
    // Safe to call even if RLC_BeginAccess failed
    void RLC_EndAccess(RLC_Surface *surface);

    // Returns true if surface is currently mapped for CUDA access
    bool RLC_IsMapped(const RLC_Surface *surface);

#ifdef __cplusplus
}
#endif

#endif // RAYLIB_CUDA_H

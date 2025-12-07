#ifndef RAYLIB_CUDA_H
#define RAYLIB_CUDA_H

#include "raylib.h"
#include <stdbool.h>

// To avoid name mangling while being compiled
#ifdef __cplusplus
extern "C"
{
#endif

    // =================================================================================
    // 1. Versions and constants
    // =================================================================================

#define RLC_VERSION_MAJOR 1
#define RLC_VERSION_MINOR 1
#define RLC_VERSION_PATCH 0

// Default bytes per pixel for RGBA8 format
// For other formats, use RLC_GetBytesPerPixel()
#define RLC_DEFAULT_BYTES_PER_PIXEL 4

    // =================================================================================
    // 2. Error Codes
    // =================================================================================

    typedef enum RLC_ERROR
    {
        RLC_OK = 0,
        RLC_ERROR_NO_CUDA_DEVICE,
        RLC_ERROR_WRONG_GPU,
        RLC_ERROR_INIT_FAILED,
        RLC_ERROR_INVALID_ARGUMENT,
        RLC_ERROR_UNSUPPORTED_FORMAT,
        RLC_ERROR_REGISTER_FAILED,
        RLC_ERROR_MAP_FAILED,
        RLC_ERROR_NOT_MAPPED,
        RLC_ERROR_ALREADY_MAPPED,
        RLC_ERROR_NULL_SURFACE
    } RLC_Error;

    // =================================================================================
    // 3. Data Types
    // =================================================================================

    typedef enum RLC_Format
    {
        RLC_FORMAT_RGBA8,  // 4 bytes/pixel, use make_uchar4(r, g, b, a)
        RLC_FORMAT_R32F,   // 4 bytes/pixel, single float (heightmaps, simulations)
        RLC_FORMAT_RGBA32F // 16 bytes/pixel, 4 floats (HDR, Physics buffers)
    } RLC_Format;

    // A Wrapper around a raylib texture + CUDA Resource
    // Note: Fields prefixed with '_' are internal - do not modify directly.
    typedef struct RLC_Surface
    {
        Texture2D texture; // Raylib texture for drawing
        int width;         // Surface width in pixels
        int height;        // Surface height in pixels
        RLC_Format format; // Track the format

        // Internal Fields (Do not modify)
        void *_cuda_res;              // CUDA Graphics resource handle
        unsigned long long _surf_obj; // CUDA surface object (valid only when mapped)
        bool _is_mapped;
        int _bytes_per_pixel; // Cached for convenience
    } RLC_Surface;

    // =================================================================================
    // 4. Library Management
    // =================================================================================

    // [DEPRECATED] Initializes CUDA and Raylib Window together
    // Use RLC_InitCUDA() instead for more control over window configuration
    // This function will be removed in v2.0
    bool RLC_Init(int width, int height, const char *title);

    // Initialize CUDA context only
    // Call this AFTER InitWindow() to allow raylib configuration
    // Returns true on success, false if no CUDA Compatible GPU is found
    // Usage:
    //  SetConfigFlags(FLAG_VSYNC_HINT | FLAG_MSAA_4X_HINT);
    //  if (!RLC_InitCUDA()) { CloseWindow(); return 1;}
    bool RLC_InitCUDA();

    // Closes CUDA resources (can call before CloseWindow if using RLC_InitCUDA)
    void RLC_CloseCUDA();

    // Closes window and cleans up all resources (for use with RLC_Init)
    void RLC_Close();

    // Return the last error code
    RLC_Error RLC_GetLastError(void);

    // Return a human-readable error message for the given error code
    const char *RLC_ErrorString(RLC_Error error);

    // =================================================================================
    // 5. Surface Management
    // =================================================================================

    // Creates a surface for CUDA rendering with RGBA8 format (default)
    // Returns a surface with _cuda_res set to NULL on failure
    // Checks with: if(surface._cuda_res == NULL) {/* handle error */}
    RLC_Surface RLC_CreateSurface(int width, int height);

    // Creates a surface with specified format
    RLC_Surface RLC_CreateSurfaceEx(int width, int height, RLC_Format format);

    // Resizes an existing surface
    // The surface must not be mapped when calling this function
    // Returns true on success and false on failure
    bool RLC_ResizeSurface(RLC_Surface *surface, int newWidth, int newHeight);

    // Returns bytes per pixel for a format
    int RLC_GetBytesPerPixel(RLC_Format format);

    // Frees the surface and associated resources
    // Safe to call with NULL or already-freed surface
    void RLC_UnloadSurface(RLC_Surface *surface);

    // =================================================================================
    // 6. Execution Pipeline
    // =================================================================================

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

    // Ends CUDA access and Synchronizes GPU
    // This is a BLOCKING call - CPU waits for GPU to complete all work
    // Must be called after RLC_BeginAccess() before using texture in with Raylib
    // Safe to call even if RLC_BeginAccess failed
    void RLC_EndAccess(RLC_Surface *surface);

    // Ends CUDA access WITHOUT synchronization (advanced usage)
    // WARNING: You must ensure GPU work is complete before drawing the texture
    // Use cudaStreamSynchronize() or cudaEventSynchronize() manually
    // Useful for overlapping CPU work with GPU rendering
    void RLC_EndAccessAsync(RLC_Surface *surface);

    // Explicitly synchronizes the GPU (call after RLC_EndAccessAsync when ready)
    void RLC_Sync(void);

    // Returns true if surface is currently mapped for CUDA access
    bool RLC_IsMapped(const RLC_Surface *surface);

    // =================================================================================
    // 7. Utility Functions
    // =================================================================================

    // Get the raylib texture from a surface (for drawing)
    Texture2D RLC_GetTexture(const RLC_Surface *surface);

    // Get surface dimensions
    int RLC_GetWidth(const RLC_Surface *surface);
    int RLC_GetHeight(const RLC_Surface *surface);

    // Get surface format
    RLC_Format RLC_GetFormat(const RLC_Surface *surface);

    // Check if surface is valid (properly initialized)
    bool RLC_IsValid(const RLC_Surface *surface);

    // Get library version
    void RLC_GetVersion(int *major, int *minor, int *patch);

#ifdef __cplusplus
}
#endif

#endif // RAYLIB_CUDA_H

/*
 * raylib_cuda.c
 * Library Implementation - handles raylib integration
 * No Windows/GL headers here to prevent conflicts
 */

#include "raylib_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ===============================================================
// Backend function declarations: Internal externs to C++ Backend
// ===============================================================

extern int rlc_backend_check();
extern void *rlc_backend_register(unsigned int id);
extern void rlc_backend_unregister(void *res);
extern unsigned long long rlc_backend_map(void *res);
extern void rlc_backend_unmap(void *res, unsigned long long surf, bool sync);
extern void rlc_backend_sync(void);
extern void rlc_backend_reset(void);

// ===============================================================
// Global State
// ===============================================================
static RLC_Error g_last_error = RLC_OK;

// ===============================================================
// Error Handling
// ===============================================================

static void rlc_set_error(RLC_Error err)
{
    g_last_error = err;
    if (err != RLC_OK)
    {
        TraceLog(LOG_WARNING, "RLC: Error set: %s", RLC_ErrorString(err));
    }
}

RLC_Error RLC_GetLastError(void)
{
    RLC_Error err = g_last_error;
    g_last_error = RLC_OK; // Clear after reading
    return err;
}

const char *RLC_ErrorString(RLC_Error error)
{
    switch (error)
    {
    case RLC_OK:
        return "No Error";
    case RLC_ERROR_NO_CUDA_DEVICE:
        return "No CUDA device found";
    case RLC_ERROR_WRONG_GPU:
        return "Wrong GPU (Intel Integrated) Detected - need discrete NVIDIA GPU";
    case RLC_ERROR_INIT_FAILED:
        return "Initialization failed";
    case RLC_ERROR_INVALID_ARGUMENT:
        return "Invalid argument";
    case RLC_ERROR_UNSUPPORTED_FORMAT:
        return "Unsupported surface format";
    case RLC_ERROR_REGISTER_FAILED:
        return "Failed to register texture with CUDA";
    case RLC_ERROR_MAP_FAILED:
        return "Failed to map resource for CUDA access";
    case RLC_ERROR_NOT_MAPPED:
        return "Surface is not mapped";
    case RLC_ERROR_ALREADY_MAPPED:
        return "Surface is already mapped";
    case RLC_ERROR_NULL_SURFACE:
        return "NULL surface pointer";
    default:
        return "Unknown error";
    }
}

// ===============================================================
// Library Management
// ===============================================================

bool RLC_InitCUDA(void)
{
    g_last_error = RLC_OK;

    // Verify window is already created
    if (!IsWindowReady())
    {
        TraceLog(LOG_ERROR, "RLC: Window must be initialized before calling RLC_InitCUDA()");
        TraceLog(LOG_ERROR, "RLC: Call InitWindow() first, then call RLC_InitCUDA()");
        rlc_set_error(RLC_ERROR_INIT_FAILED);
        return false;
    }

    // Check CUDA availability
    int result = rlc_backend_check();
    if (result != 0)
    {
        if (result == 1)
        {
            rlc_set_error(RLC_ERROR_NO_CUDA_DEVICE);
            TraceLog(LOG_ERROR, "RLC: No CUDA device found");
        }
        else if (result == 2)
        {
            rlc_set_error(RLC_ERROR_WRONG_GPU);
            TraceLog(LOG_ERROR, "RLC: Intel GPU detected - need discrete NVIDIA GPU");
        }
        return false;
    }
    TraceLog(LOG_INFO, "RLC: Raylib-CUDA v%d.%d.%d initialized successfully",
             RLC_VERSION_MAJOR, RLC_VERSION_MINOR, RLC_VERSION_PATCH);
    return true;
}

void RLC_CloseCUDA(void)
{
    // Reset CUDA device (cleans up any remaining resources)
    rlc_backend_reset();
    TraceLog(LOG_INFO, "RLC: CUDA shutdown complete");
}

// DEPRECATED - Kept for backward compatibility
bool RLC_Init(int width, int height, const char *title)
{
    // Deprecation Warning
    TraceLog(LOG_WARNING, "RLC: RLC_Init() is deprecated. Use InitWindow() + RLC_InitCUDA() instead");

    g_last_error = RLC_OK;

    // Initialize raylib window
    InitWindow(width, height, title);

    if (!IsWindowReady())
    {
        TraceLog(LOG_ERROR, "RLC: Failed to create window");
        return false;
    }

    // Check CUDA availability
    int result = rlc_backend_check();
    if (result != 0)
    {
        if (result == 1)
        {
            rlc_set_error(RLC_ERROR_NO_CUDA_DEVICE);
            TraceLog(LOG_ERROR, "RLC: No CUDA device found");
        }
        else if (result == 2)
        {
            rlc_set_error(RLC_ERROR_WRONG_GPU);
            TraceLog(LOG_ERROR, "RLC: Intel GPU detected - need discrete NVIDIA GPU");
        }

        // Clean up the window
        CloseWindow();
        return false;
    }

    TraceLog(LOG_INFO, "RLC: Raylib-CUDA v%d.%d.%d initialized successfully.",
             RLC_VERSION_MAJOR, RLC_VERSION_MINOR, RLC_VERSION_PATCH);
    return true;
}

void RLC_Close()
{
    RLC_CloseCUDA();
    CloseWindow();
    TraceLog(LOG_INFO, "RLC: Shutdown complete");
}

// ===============================================================
// Surface Management
// ===============================================================

int RLC_GetBytesPerPixel(RLC_Format format)
{
    switch (format)
    {
    case RLC_FORMAT_RGBA8:
        return 4;
    case RLC_FORMAT_R32F:
        return 4;
    case RLC_FORMAT_RGBA32F:
        return 16;
    default:
        return 4;
    }
}

static int rlc_format_to_raylib(RLC_Format format)
{
    switch (format)
    {
    case RLC_FORMAT_RGBA8:
        return PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
    case RLC_FORMAT_R32F:
        return PIXELFORMAT_UNCOMPRESSED_R32;
    case RLC_FORMAT_RGBA32F:
        return PIXELFORMAT_UNCOMPRESSED_R32G32B32A32;
    default:
        return PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
    }
}

RLC_Surface RLC_CreateSurface(int width, int height)
{
    return RLC_CreateSurfaceEx(width, height, RLC_FORMAT_RGBA8);
}

RLC_Surface RLC_CreateSurfaceEx(int width, int height, RLC_Format format)
{
    RLC_Surface surf = {0};

    // Validate Dimensions
    if (width <= 0 || height <= 0)
    {
        TraceLog(LOG_ERROR, "RLC: Invalid surface dimensions: %dx%d", width,
                 height);
        rlc_set_error(RLC_ERROR_INVALID_ARGUMENT);
        return surf;
    }

    // Validate format
    if (format < RLC_FORMAT_RGBA8 || format > RLC_FORMAT_RGBA32F)
    {
        TraceLog(LOG_ERROR, "RLC: Invalid surface format: %d", format);
        rlc_set_error(RLC_ERROR_UNSUPPORTED_FORMAT);
        return surf;
    }

    surf.width = width;
    surf.height = height;
    surf.format = format;
    surf._is_mapped = false;
    surf._surf_obj = 0;
    surf._bytes_per_pixel = RLC_GetBytesPerPixel(format);

    // Create image with appropriate format
    int raylib_format = rlc_format_to_raylib(format);

    Image img = {0};
    img.width = width;
    img.height = height;
    img.mipmaps = 1;
    img.format = raylib_format;

    // Allocate and zero the pixel data
    size_t data_size = (size_t)width * height * surf._bytes_per_pixel;
    img.data = RL_CALLOC(1, data_size);

    if (img.data == NULL)
    {
        TraceLog(LOG_ERROR, "RLC: Failed to allocate image data");
        rlc_set_error(RLC_ERROR_REGISTER_FAILED);
        return surf;
    }

    surf.texture = LoadTextureFromImage(img);
    UnloadImage(img);

    if (surf.texture.id == 0)
    {
        TraceLog(LOG_ERROR, "RLC: Failed to create texture");
        rlc_set_error(RLC_ERROR_REGISTER_FAILED);
        return surf;
    }

    // Set point filtering (no interpolation for pixel-perfect rendering)
    SetTextureFilter(surf.texture, TEXTURE_FILTER_POINT);

    // CUDA: Register
    surf._cuda_res = rlc_backend_register(surf.texture.id);
    if (surf._cuda_res == NULL)
    {
        TraceLog(LOG_ERROR, "RLC: Failed to register texture with CUDA");
        UnloadTexture(surf.texture);
        rlc_set_error(RLC_ERROR_REGISTER_FAILED);
        memset(&surf, 0, sizeof(surf));
        return surf;
    }

    TraceLog(LOG_INFO, "RLC: Created surface %dx%d (format=%d, bpp=%d)",
             width, height, format, surf._bytes_per_pixel);
    return surf;
}

bool RLC_ResizeSurface(RLC_Surface *surface, int newWidth, int newHeight)
{
    if (surface == NULL)
    {
        rlc_set_error(RLC_ERROR_NULL_SURFACE);
        return false;
    }

    if (newWidth <= 0 || newHeight <= 0)
    {
        rlc_set_error(RLC_ERROR_INVALID_ARGUMENT);
        return false;
    }

    if (surface->_is_mapped)
    {
        TraceLog(LOG_ERROR, "RLC: Cannot resize mapped surface - call RLC_EndAccess first");
        rlc_set_error(RLC_ERROR_ALREADY_MAPPED);
        return false;
    }

    // Same size -> Do nothing
    if (surface->width == newWidth && surface->height == newHeight)
    {
        return true;
    }

    // Save format before cleanup
    RLC_Format format = surface->format;

    // Unregister from CUDA
    if (surface->_cuda_res != NULL)
    {
        rlc_backend_unregister(surface->_cuda_res);
        surface->_cuda_res = NULL;
    }

    // Unload old texture
    if (surface->texture.id != 0)
    {
        UnloadTexture(surface->texture);
    }

    // Create new surface with same format
    RLC_Surface newSurf = RLC_CreateSurfaceEx(newWidth, newHeight, format);

    if (newSurf._cuda_res == NULL)
    {
        // Failed - surface is now invalid
        memset(surface, 0, sizeof(*surface));
        return false;
    }

    // Copy new surface data to old surface pointer
    *surface = newSurf;

    TraceLog(LOG_INFO, "RLC: Resized surface to %dx%d", newWidth, newHeight);
    return true;
}

void RLC_UnloadSurface(RLC_Surface *surface)
{
    if (surface == NULL)
    {
        return;
    }

    // End access if still mapped
    if (surface->_is_mapped)
    {
        TraceLog(LOG_WARNING,
                 "RLC: Surface was still mapped during unload - unmapping");
        RLC_EndAccess(surface);
    }

    // Unregister from CUDA
    if (surface->_cuda_res != NULL)
    {
        rlc_backend_unregister(surface->_cuda_res);
        surface->_cuda_res = NULL;
    }

    // Unload raylib texture
    if (surface->texture.id != 0)
    {
        UnloadTexture(surface->texture);
    }

    // Clear the struct
    memset(surface, 0, sizeof(*surface));

    TraceLog(LOG_DEBUG, "RLC: Surface unloaded");
}

// ===============================================================
// Execution Pipeline
// ===============================================================

unsigned long long RLC_BeginAccess(RLC_Surface *surface)
{
    if (surface == NULL)
    {
        rlc_set_error(RLC_ERROR_NULL_SURFACE);
        return 0;
    }

    if (surface->_cuda_res == NULL)
    {
        rlc_set_error(RLC_ERROR_NULL_SURFACE);
        TraceLog(LOG_ERROR, "RLC: BeginAccess on invalid surface");
        return 0;
    }

    if (surface->_is_mapped)
    {
        rlc_set_error(RLC_ERROR_ALREADY_MAPPED);
        TraceLog(LOG_WARNING,
                 "RLC: Surface already mapped - returning existing handle");
        return surface->_surf_obj;
    }

    surface->_surf_obj = rlc_backend_map(surface->_cuda_res);

    if (surface->_surf_obj == 0)
    {
        rlc_set_error(RLC_ERROR_MAP_FAILED);
        TraceLog(LOG_ERROR, "RLC: Failed to map surface");
        return 0;
    }
    surface->_is_mapped = true;
    return surface->_surf_obj;
}

void RLC_EndAccess(RLC_Surface *surface)
{
    if (surface == NULL || surface->_cuda_res == NULL)
    {
        return;
    }

    if (!surface->_is_mapped)
    {
        // Idempotent - Safe to call multiple times
        return;
    }

    rlc_backend_unmap(surface->_cuda_res, surface->_surf_obj, true); // sync = true
    surface->_surf_obj = 0;
    surface->_is_mapped = false;
}

void RLC_EndAccessAsync(RLC_Surface *surface)
{
    if (surface == NULL || surface->_cuda_res == NULL)
    {
        return;
    }

    if (!surface->_is_mapped)
    {
        // Idempotent - Safe to call multiple times
        return;
    }

    rlc_backend_unmap(surface->_cuda_res, surface->_surf_obj, false); // sync = false
    surface->_surf_obj = 0;
    surface->_is_mapped = false;
}

void RLC_Sync(void)
{
    rlc_backend_sync();
}

bool RLC_IsMapped(const RLC_Surface *surface)
{
    if (surface == NULL)
        return false;
    return surface->_is_mapped;
}

Texture2D RLC_GetTexture(const RLC_Surface *surface)
{
    if (surface == NULL)
    {
        Texture2D empty = {0};
        return empty;
    }
    return surface->texture;
}

int RLC_GetWidth(const RLC_Surface *surface)
{
    return (surface != NULL) ? surface->width : 0;
}

int RLC_GetHeight(const RLC_Surface *surface)
{
    return (surface != NULL) ? surface->height : 0;
}

RLC_Format RLC_GetFormat(const RLC_Surface *surface)
{
    return (surface != NULL) ? surface->format : RLC_FORMAT_RGBA8;
}

bool RLC_IsValid(const RLC_Surface *surface)
{
    return (surface != NULL &&
            surface->_cuda_res != NULL &&
            surface->texture.id != 0);
}

void RLC_GetVersion(int *major, int *minor, int *patch)
{
    if (major)
        *major = RLC_VERSION_MAJOR;
    if (minor)
        *minor = RLC_VERSION_MINOR;
    if (patch)
        *patch = RLC_VERSION_PATCH;
}
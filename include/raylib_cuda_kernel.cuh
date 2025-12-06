#ifndef RAYLIB_CUDA_KERNEL_CUH
#define RAYLIB_CUDA_KERNEL_CUH

/*
 * raylib_cuda_kernel.cuh
 * Helper definitions for writing CUDA kernels with raylib_cuda
 * Include this in your .cu files
 */

#include <cuda_runtime.h>

// =====================================================
// RGBA8 Pixel Writing Helpers (4 bytes per pixel)
// =====================================================

// Write an RGBA pixel to a surface
// x, y: pixel coordinates
// r, g, b, a: color components (0-255)
__device__ inline void rlcWritePixel(cudaSurfaceObject_t surf, int x, int y,
                                     unsigned char r, unsigned char g,
                                     unsigned char b, unsigned char a = 255)
{
    uchar4 pixel = make_uchar4(r, g, b, a);
    surf2Dwrite(pixel, surf, x * 4, y);
}

// Write an RGBA pixel using float colors (0.0 - 1.0)
__device__ inline void rlcWritePixelF(cudaSurfaceObject_t surf, int x, int y,
                                      float r, float g, float b,
                                      float a = 1.0f)
{
    uchar4 pixel =
        make_uchar4((unsigned char)(__saturatef(r) * 255.0f), (unsigned char)(__saturatef(g) * 255.0f),
                    (unsigned char)(__saturatef(b) * 255.0f), (unsigned char)(__saturatef(a) * 255.0f));
    surf2Dwrite(pixel, surf, x * 4, y);
}

// Read an RGBA pixel from a surface
__device__ inline uchar4 rlcReadPixel(cudaSurfaceObject_t surf, int x, int y)
{
    uchar4 pixel;
    surf2Dread(&pixel, surf, x * 4, y);
    return pixel;
}

// =====================================================
// R32F Pixel Writing Helpers (Single float, 4 bytes per pixel)
// =====================================================

// Write a single float value
__device__ inline void rlcWriteFloat(cudaSurfaceObject_t surf, int x, int y, float value)
{
    surf2Dwrite(value, surf, x * sizeof(float), y);
}

// Read a single float value
__device__ inline float rlcReadFloat(cudaSurfaceObject_t surf, int x, int y)
{
    float value;
    surf2Dread(&value, surf, x * sizeof(float), y);
    return value;
}
// =====================================================
// RGBA32F Pixel Writing Helpers (4 floats, 16 bytes per pixel)
// =====================================================

// Write 4 float values
__device__ inline void rlcWriteFloat4(cudaSurfaceObject_t surf, int x, int y, float r, float g, float b, float a = 1.0f)
{
    float4 pixel = make_float4(r, g, b, a);
    surf2Dwrite(pixel, surf, x * sizeof(float4), y);
}

// Read 4 float values
__device__ inline float4 rlcReadFloat4(cudaSurfaceObject_t surf, int x, int y)
{
    float4 pixel;
    surf2Dread(&pixel, surf, x * sizeof(float4), y);
    return pixel;
}

// =====================================================
// Bounds-Checked Variants (Optional safety)
// =====================================================

__device__ inline bool rlcInBounds(int x, int y, int width, int height)
{
    return (x >= 0 && x < width && y >= 0 && y < height);
}

__device__ inline void rlcWritePixelSafe(cudaSurfaceObject_t surf, int x, int y, int width, int height, unsigned char r, unsigned char g, unsigned char b, unsigned char a = 255)
{
    if (rlcInBounds(x, y, width, height))
    {
        rlcWritePixel(surf, x, y, r, g, b, a);
    }
}

#endif // RAYLIB_CUDA_KERNEL_CUH

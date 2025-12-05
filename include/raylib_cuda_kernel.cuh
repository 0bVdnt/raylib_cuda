#ifndef RAYLIB_CUDA_KERNEL_CUH
#define RAYLIB_CUDA_KERNEL_CUH

/*
 * raylib_cuda_kernel.h
 * Helper definitions for writing CUDA kernels with raylib_cuda
 * Include this in your .cu files
 */

#include <cuda_runtime.h>

// =====================================================
// Pixel Writing Helpers
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
        make_uchar4((unsigned char)(r * 255.0f), (unsigned char)(g * 255.0f),
                    (unsigned char)(b * 255.0f), (unsigned char)(a * 255.0f));
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
// Common Kernel Pattern
// =====================================================

/*
Example kernel:

__global__ void myKernel(cudaSurfaceObject_t surf, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Calculate color
    unsigned char r = (x * 255) / width;
    unsigned char g = (y * 255) / height;
    unsigned char b = 128;

    rlcWritePixel(surf, x, y, r, g, b);
}

Usage:

    unsigned long long surfObj = RLC_BeginAccess(&surface);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    myKernel<<<grid, block>>>((cudaSurfaceObject_t)surfObj, width, height);

    RLC_EndAccess(&surface);
*/

#endif // RAYLIB_CUDA_KERNEL_CUH

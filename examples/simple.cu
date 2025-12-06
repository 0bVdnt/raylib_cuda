#include "raylib_cuda.h"
#include <cstdio>

// Simple CUDA kernel - fills screen with gradient
__global__ void testKernel(cudaSurfaceObject_t surf, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    unsigned char r = (x * 255) / width;
    unsigned char g = (y * 255) / height;
    unsigned char b = 128;

    uchar4 pixel = make_uchar4(r, g, b, 255);
    surf2Dwrite(pixel, surf, x * 4, y);
}

int main()
{
    const int W = 800, H = 600;

    // Initialize
    if (!RLC_Init(W, H, "raylib_cuda Test"))
    {
        printf("Failed to initialize!\n");
        return 1;
    }

    // Create CUDA surface
    RLC_Surface surf = RLC_CreateSurface(W, H);
    if (surf._cuda_res == NULL)
    {
        printf("Failed to create surface!\n");
        RLC_Close();
        return 1;
    }

    printf("Success! Press ESC to exit.\n");

    // Kernel config
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);

    SetTargetFPS(60);

    while (!WindowShouldClose())
    {
        // CUDA rendering
        unsigned long long so = RLC_BeginAccess(&surf);
        if (so)
        {
            testKernel<<<grid, block>>>((cudaSurfaceObject_t)so, W, H);
        }
        RLC_EndAccess(&surf);

        // Display
        BeginDrawing();
        ClearBackground(BLACK);
        DrawTexture(surf.texture, 0, 0, WHITE);
        DrawFPS(10, 10);
        DrawText("CUDA + Raylib Working!", 10, 40, 20, WHITE);
        EndDrawing();
    }

    // Cleanup
    RLC_UnloadSurface(&surf);
    RLC_Close();

    return 0;
}
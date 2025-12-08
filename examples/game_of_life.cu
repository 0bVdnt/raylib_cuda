// examples/game_of_life.cu
//
// Canonical example of using raylib_cuda:
// - InitWindow() -> RLC_InitCUDA()
// - Create RLC_Surface
// - Map with RLC_BeginAccess()
// - Launch CUDA kernels using cudaSurfaceObject_t
// - Unmap with RLC_EndAccess()
// - Draw using raylib Texture2D from RLC_GetTexture()

#include "raylib_cuda.h"
#include "raylib_cuda_kernel.cuh"
#include <stdio.h>
#include <time.h>

// Simple hash-based pseudo-random generator in [0, 1]
__device__ float hash01(int x, int y, unsigned int seed)
{
    unsigned int h = seed;
    h ^= (unsigned int)x * 374761393u;
    h = (h << 5) | (h >> 27);
    h ^= (unsigned int)y * 668265263u;
    h *= 0x27d4eb2du;
    // Keep the lower 24 bits and normalize
    return (h & 0xFFFFFFu) / 16777215.0f;
}

// Initialize random Game of Life grid
__global__ void initRandom(cudaSurfaceObject_t surf,
                           int width, int height,
                           unsigned int seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    unsigned char alive = (hash01(x, y, seed) > 0.85f) ? 255 : 0;
    rlcWritePixel(surf, x, y, alive, alive, alive);
}

// One step
__global__ void stepLife(cudaSurfaceObject_t src,
                         cudaSurfaceObject_t dst,
                         int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Count neighbors (with toroidal wrap-up)
    int neighbors = 0;
    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            if (dx == 0 && dy == 0)
                continue;

            int nx = (x + dx + width) % width;
            int ny = (y + dy + height) % height;

            uchar4 pixel = rlcReadPixel(src, nx, ny);
            if (pixel.x > 128)
                neighbors++;
        }
    }

    uchar4 current = rlcReadPixel(src, x, y);
    bool alive = current.x > 128;

    // Conway's rules
    bool nextAlive = alive ? (neighbors == 2 || neighbors == 3)
                           : (neighbors == 3);

    unsigned char val = nextAlive ? 255 : 0;
    rlcWritePixel(dst, x, y, val, val, val);
}

int main(void)
{
    const int WIDTH = 800;
    const int HEIGHT = 600;

    SetConfigFlags(FLAG_VSYNC_HINT);
    InitWindow(WIDTH, HEIGHT, "raylib_cuda - Game of Life");
    SetTargetFPS(60);

    // Initialize CUDA interop AFTER InitWindow()
    if (!RLC_InitCUDA())
    {
        RLC_Error err = RLC_GetLastError();
        printf("RLC_InitCUDA failed: %s (code %d)\n",
               RLC_ErrorString(err), err);
        CloseWindow();
        return 1;
    }

    // Double buffer for simulation
    RLC_Surface surfaceA = RLC_CreateSurface(WIDTH, HEIGHT);
    RLC_Surface surfaceB = RLC_CreateSurface(WIDTH, HEIGHT);

    if (!RLC_IsValid(&surfaceA) || !RLC_IsValid(&surfaceB))
    {
        RLC_Error err = RLC_GetLastError();
        printf("Surface creation failed: %s (code %d)\n",
               RLC_ErrorString(err), err);

        RLC_UnloadSurface(&surfaceA);
        RLC_UnloadSurface(&surfaceB);
        RLC_Close(); // Closes CUDA + Window
        return 1;
    }

    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x,
              (HEIGHT + block.y - 1) / block.y);

    // Initialize with random state
    cudaSurfaceObject_t surf =
        (cudaSurfaceObject_t)RLC_BeginAccess(&surfaceA);

    if (surf != 0)
    {
        initRandom<<<grid, block>>>(surf, WIDTH, HEIGHT, (unsigned int)time(NULL));
        RLC_EndAccess(&surfaceA);
    }
    else
    {
        RLC_Error err = RLC_GetLastError();
        printf("Initial RLC_BeginAccess failed: %s (code %d)",
               RLC_ErrorString(err), err);
    }

    RLC_Surface *current = &surfaceA;
    RLC_Surface *next = &surfaceB;

    int generation = 0;
    bool paused = false;

    while (!WindowShouldClose())
    {
        // Controls
        if (IsKeyPressed(KEY_SPACE))
            paused = !paused;

        if (IsKeyPressed(KEY_R))
        {
            cudaSurfaceObject_t s =
                (cudaSurfaceObject_t)RLC_BeginAccess(current);
            if (s != 0)
            {
                initRandom<<<grid, block>>>(s, WIDTH, HEIGHT,
                                            (unsigned int)time(NULL));
                RLC_EndAccess(current);
                generation = 0;
            }
            else
            {
                RLC_Error err = RLC_GetLastError();
                printf("Reset RLC_BeginAccess failed: %s (code %d)\n",
                       RLC_ErrorString(err), err);
            }
        }

        // Simulation step
        if (!paused)
        {
            cudaSurfaceObject_t srcSurf =
                (cudaSurfaceObject_t)RLC_BeginAccess(current);
            cudaSurfaceObject_t dstSurf =
                (cudaSurfaceObject_t)RLC_BeginAccess(next);
            if (srcSurf != 0 && dstSurf != 0)
            {

                stepLife<<<grid, block>>>(srcSurf, dstSurf, WIDTH, HEIGHT);
                RLC_EndAccess(current);
                RLC_EndAccess(next);

                // Swap buffers
                RLC_Surface *temp = current;
                current = next;
                next = temp;

                generation++;
            }
            else
            {
                RLC_Error err = RLC_GetLastError();
                printf("Step RLC_BeginAccess failed: %s (code %d)",
                       RLC_ErrorString(err), err);

                if (srcSurf != 0)
                    RLC_EndAccess(current);
                if (dstSurf != 0)
                    RLC_EndAccess(next);
            }
        }

        BeginDrawing();
            ClearBackground(BLACK);
            DrawTexture(RLC_GetTexture(current), 0, 0, WHITE);
            DrawText(TextFormat("Generation: %d", generation),
                                10, 10, 20, GREEN);
            DrawText(paused ? "PAUSED (Space to resume)" 
                            : "Space: Pause | R: Reset",
                            10, 35, 16, GRAY);
            DrawFPS(10, HEIGHT - 30);
        EndDrawing();
    }

    RLC_UnloadSurface(&surfaceA);
    RLC_UnloadSurface(&surfaceB);
    RLC_Close(); // Closes CUDA + Window
    return 0;
}
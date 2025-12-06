/*
 * test_visual.cu
 * Visual/interactive tests for raylib_cuda
 * These require manual verification
 */

#include "raylib_cuda.h"
#include "raylib_cuda_kernel.h"
#include <cstdio>
#include <cmath>

// =====================================================
// Kernels
// =====================================================

__global__ void gradientKernel(cudaSurfaceObject_t surf, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    unsigned char r = (x * 255) / width;
    unsigned char g = (y * 255) / height;
    unsigned char b = 128;
    
    rlcWritePixel(surf, x, y, r, g, b);
}

__global__ void checkerboardKernel(cudaSurfaceObject_t surf, int width, int height, int squareSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int cx = x / squareSize;
    int cy = y / squareSize;
    bool isWhite = (cx + cy) % 2 == 0;
    
    unsigned char c = isWhite ? 255 : 0;
    rlcWritePixel(surf, x, y, c, c, c);
}

__global__ void animatedKernel(cudaSurfaceObject_t surf, int width, int height, float time)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float fx = (float)x / width;
    float fy = (float)y / height;
    
    float r = sinf(fx * 6.28f + time) * 0.5f + 0.5f;
    float g = sinf(fy * 6.28f + time * 1.3f) * 0.5f + 0.5f;
    float b = sinf((fx + fy) * 6.28f + time * 0.7f) * 0.5f + 0.5f;
    
    rlcWritePixelF(surf, x, y, r, g, b);
}

__global__ void mandelbrotKernel(cudaSurfaceObject_t surf, int width, int height, 
                                  float centerX, float centerY, float zoom)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px >= width || py >= height) return;
    
    float x0 = centerX + (px - width / 2.0f) / (width * zoom);
    float y0 = centerY + (py - height / 2.0f) / (height * zoom);
    
    float x = 0, y = 0;
    int iter = 0;
    const int maxIter = 256;
    
    while (x * x + y * y <= 4 && iter < maxIter) {
        float xtemp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = xtemp;
        iter++;
    }
    
    unsigned char c = (iter == maxIter) ? 0 : (unsigned char)(iter % 256);
    rlcWritePixel(surf, px, py, c, c / 2, c * 2);
}

// =====================================================
// Visual Tests
// =====================================================

void run_visual_test()
{
    const int WIDTH = 800;
    const int HEIGHT = 600;
    
    if (!RLC_Init(WIDTH, HEIGHT, "Visual Test - Press 1-4 to switch, ESC to exit")) {
        printf("Failed to initialize\n");
        return;
    }
    
    SetTargetFPS(60);
    
    RLC_Surface surf = RLC_CreateSurface(WIDTH, HEIGHT);
    if (surf._cuda_res == NULL) {
        printf("Failed to create surface\n");
        RLC_Close();
        return;
    }
    
    dim3 block(16, 16);
    dim3 grid((WIDTH + 15) / 16, (HEIGHT + 15) / 16);
    
    int currentTest = 1;
    float time = 0;
    float zoom = 0.5f;
    float centerX = -0.5f, centerY = 0.0f;
    
    printf("\nVisual Test Controls:\n");
    printf("  1 - Gradient\n");
    printf("  2 - Checkerboard\n");
    printf("  3 - Animated waves\n");
    printf("  4 - Mandelbrot (use arrows to pan, +/- to zoom)\n");
    printf("  ESC - Exit\n\n");
    
    while (!WindowShouldClose())
    {
        // Input
        if (IsKeyPressed(KEY_ONE)) currentTest = 1;
        if (IsKeyPressed(KEY_TWO)) currentTest = 2;
        if (IsKeyPressed(KEY_THREE)) currentTest = 3;
        if (IsKeyPressed(KEY_FOUR)) currentTest = 4;
        
        // Mandelbrot controls
        if (currentTest == 4) {
            float panSpeed = 0.5f / zoom;
            if (IsKeyDown(KEY_LEFT)) centerX -= panSpeed * GetFrameTime();
            if (IsKeyDown(KEY_RIGHT)) centerX += panSpeed * GetFrameTime();
            if (IsKeyDown(KEY_UP)) centerY -= panSpeed * GetFrameTime();
            if (IsKeyDown(KEY_DOWN)) centerY += panSpeed * GetFrameTime();
            if (IsKeyDown(KEY_EQUAL)) zoom *= 1.02f;  // + key
            if (IsKeyDown(KEY_MINUS)) zoom /= 1.02f;
        }
        
        time += GetFrameTime();
        
        // CUDA rendering
        unsigned long long surfObj = RLC_BeginAccess(&surf);
        if (surfObj != 0) {
            cudaSurfaceObject_t so = (cudaSurfaceObject_t)surfObj;
            
            switch (currentTest) {
                case 1:
                    gradientKernel<<<grid, block>>>(so, WIDTH, HEIGHT);
                    break;
                case 2:
                    checkerboardKernel<<<grid, block>>>(so, WIDTH, HEIGHT, 32);
                    break;
                case 3:
                    animatedKernel<<<grid, block>>>(so, WIDTH, HEIGHT, time);
                    break;
                case 4:
                    mandelbrotKernel<<<grid, block>>>(so, WIDTH, HEIGHT, centerX, centerY, zoom);
                    break;
            }
        }
        RLC_EndAccess(&surf);
        
        // Raylib rendering
        BeginDrawing();
            ClearBackground(BLACK);
            DrawTexture(surf.texture, 0, 0, WHITE);
            
            // UI overlay
            DrawRectangle(5, 5, 200, 80, Fade(BLACK, 0.7f));
            DrawText(TextFormat("FPS: %d", GetFPS()), 10, 10, 20, GREEN);
            DrawText(TextFormat("Test: %d", currentTest), 10, 35, 20, WHITE);
            
            if (currentTest == 4) {
                DrawText(TextFormat("Zoom: %.2f", zoom), 10, 60, 16, GRAY);
            }
        EndDrawing();
    }
    
    RLC_UnloadSurface(&surf);
    RLC_Close();
}

// =====================================================
// Main
// =====================================================

int main()
{
    printf("========================================\n");
    printf("  raylib_cuda Visual Tests\n");
    printf("========================================\n");
    
    run_visual_test();
    
    printf("\nVisual test complete.\n");
    return 0;
}
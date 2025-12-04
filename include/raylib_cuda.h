#ifndef RAYLIB_CUDA_H
#define RAYLIB_CUDA_H

#include "raylib.h"
#include <stdbool.h>

// To avoid name mangling while being compiled
#ifdef __cplusplus
extern "C" {
#endif
// =====================================================
// 1. Data Types
// =====================================================

// A Wrapper around a raylib texture + CUDA Resource
typedef struct RLC_Surface {
    Texture2D texture; // Raylib texture for drawing
    int width;
    int height;

    // Internal use (Opaque Handles)
    void *_cuda_res;
    unsigned long long _surf_obj;
} RLC_Surface;

// =====================================================
// 2. Library Management
// =====================================================

// Initializes cuda and checks for correct GPU
// Return true on success and false if No CUDA enabled GPU is found
bool RLC_Init(int width, int height, const char *title);

// Closes window and cleans up
void RLC_Close();

// =====================================================
// 3. Surface Management
// =====================================================

// Creates a texture ready for CUDA Writing
RLC_Surface RLC_CreateSurface(int width, int height);

// Frees the surface
void RLC_UnloadSurface(RLC_Surface *surface);

// =====================================================
// 4. Execution Pipeline
// =====================================================

// Locks the texture and return a CUDA Surface Object
// Pass this `unsigned long long` to the kernel.
unsigned long long RLC_BeginAccess(RLC_Surface *surface);

// Locks the texture and synchronizes the GPU
void RLC_EndAccess(RLC_Surface* surface);

#ifdef __cplusplus
}
#endif

#endif // RAYLIB_CUDA_H
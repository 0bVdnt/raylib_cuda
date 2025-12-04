/*
* Library Implementation of raylib_cuda.h
* Handles raylib. No Windows/GL headers allowed here to prevent conflicts.
*/

// System Headers
#include <stdio.h>
#include <stdlib.h>

// Include Raylib
#include "../include/raylib_cuda.h"

// Backend declarations: Internal externs to C++ Backend
extern int rlc_backend_check();
extern void *rlc_backend_register(unsigned int *id);
extern void rlc_backend_unregister(void *res);
extern unsigned long long rlc_backend_map(void *res);
extern void rlc_backend_unmap(void *res, unsigned long long surf);

// Implementation

bool RLC_Init(int width, int height, const char *title) {
    InitWindow(width, height, title);
    
    // Check CUDA
    if (rlc_backend_check() != 0) {
        TraceLog(LOG_ERROR, "RLC: CUDA Device not Found or Compatible !");
        return false;
    }

    TraceLog(LOG_INFO, "RLC: Raylib-CUDA initialized successfully.");
    return true;
}

void RLC_Close() {
    CloseWindow();
}

RLC_Surface RLC_CreateSurface(int width, int height) {
    RLC_Surface surf = {0};
    surf.width = width;
    surf.height = height;

    // Raylib: Create texture
    Image img = GenImageColor(width, height, BLACK);
    surf.texture = LoadTextureFromImage(img);
    SetTextureFilter(surf.texture, TEXTURE_FILTER_POINT);
    UnloadImage(img);

    // CUDA: Register (Backend handles GL types)
    surf._cuda_res = rlc_backend_register(surf.texture.id);
    return surf;
}

void RLC_UnloadSurface(RLC_Surface *surface) {
    if (surface->_cuda_res)
        rlc_backend_unregister(surface->_cuda_res);
    UnloadTexture(surface->texture);
}

unsigned long long RLC_BeginAccess(RLC_Surface *surface) {
    if (!surface->_cuda_res)
        return 0;
    surface->_surf_obj = rlc_backend_map(surface->_cuda_res);
    return surface->_surf_obj;
}

void RLC_EndAccess(RLC_Surface *surface) {
    if (!surface->_cuda_res)
        return;
    rlc_backend_unmap(surface->_cuda_res, surface->_surf_obj);
    surface->_surf_obj = 0;
}
# raylib_cuda

<div align="center">

![Version](https://img.shields.io/badge/version-1.1.1-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900.svg?logo=nvidia)
![raylib](https://img.shields.io/badge/raylib-5.5+-red.svg)
![Platforms](https://img.shields.io/badge/platforms-Windows%20|%20Linux-lightgrey.svg)

**A CUDA-OpenGL interop library for raylib**

*Write GPU-accelerated graphics with CUDA kernels and render them seamlessly with raylib — zero CPU round-trip required.*

[Features](#features) •
[How It Works](#how-it-works) •
[Installation](#installation) •
[Quick Start](#quick-start) •
[API Reference](#api-reference) •
[Examples](#examples) •
[Troubleshooting](#troubleshooting)

</div>

---

## Overview

**raylib_cuda** bridges NVIDIA's CUDA parallel computing platform with the simplicity of raylib. Write CUDA kernels to generate or process images on the GPU, then display them directly using raylib's rendering functions.

Normally, getting GPU-computed pixels onto a raylib texture means allocating CPU memory, copying data back from the GPU, then uploading to OpenGL. This round-trip through system memory is slow and defeats the purpose of GPU computation for real-time visuals. **raylib_cuda** eliminates this bottleneck entirely — the data never leaves the GPU.
<div align="center">
<pre>
┌──────────────────────────────────────────────────────────────────────┐
│                                  GPU                                 │
│                                                                      │
│   CUDA Kernel ──writes to──► OpenGL Texture -─► drawn by ──► raylib  │
│                              (shared memory)                         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
No CPU copy. No download. No upload.
</pre>
</div>
### Use Cases

- **Real-time procedural generation** — Fractals, noise, terrain heightmaps
- **GPU-accelerated simulations** — Particle systems, fluid dynamics, cellular automata
- **Image processing pipelines** — Filters, convolutions, computer vision
- **Scientific visualization** — Heat maps, field visualizations, data plotting
- **Game effects** — Dynamic textures, GPU-driven animations

---

## Features

- **Zero-copy rendering** — CUDA writes directly to OpenGL textures
- **Multiple pixel formats** — RGBA8, R32F, RGBA32F for different use cases
- **Simple, minimal API** — `BeginAccess` / launch kernel / `EndAccess` / draw
- **Double-buffering support** — Swap between surfaces for simulation patterns
- **Surface resizing** — Resize without leaking resources
- **Async support** — Optional non-blocking GPU synchronization
- **Kernel helpers** — `rlcWritePixel()`, `rlcReadPixel()`, bounds-checked variants
- **Hybrid GPU handling** — Detects Intel integrated GPUs with actionable guidance
- **Safe by default** — Comprehensive error handling and validation
- **Easy integration** — CMake-based, auto-fetches raylib if needed
- **Cross-platform** — Windows and Linux

---

## How It Works

The library manages ownership of an OpenGL texture between two GPU subsystems:

```
1. RLC_CreateSurface()     Create a raylib texture and register it with CUDA
                           (one-time setup)

2. RLC_BeginAccess()       Map the texture for CUDA access
                           Returns a cudaSurfaceObject_t for kernel use
          │
          ▼
   Launch CUDA kernels     Write pixels using surf2Dwrite / helper functions
          │
          ▼
3. RLC_EndAccess()         Unmap the texture, synchronize GPU
                           Texture is now safe for raylib to draw

4. DrawTexture()           Raylib renders the texture — pixels are already there
```

Internally, the library separates concerns into two layers:

- **`raylib_cuda.c`** — Pure C public API. Manages surfaces, lifecycle, errors, and raylib integration. Contains no GPU or GL headers.
- **`raylib_cuda_backend.cu`** — CUDA/C++ backend. Handles all `cudaGraphicsGL*` interop calls and platform-specific GL includes. Exposed to the C layer via `extern "C"` functions.

This separation prevents header conflicts between Windows.h, OpenGL, and CUDA — a real problem on Windows builds.

---

## Requirements

### Minimum Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **NVIDIA GPU** | Compute Capability 5.2 (Maxwell) | Compute Capability 7.5+ (Turing/Ampere/Ada) |
| **CUDA Toolkit** | 11.0 | 12.x |
| **CMake** | 3.18 | 3.24+ (for `native` arch detection) |
| **C Compiler** | C99 compatible | GCC, Clang, or MSVC |
| **C++ Compiler** | C++11 compatible | Bundled with CUDA Toolkit |
| **raylib** | 5.0 | 5.5 (fetched automatically) |

### Supported Platforms

| Platform | Status | Notes |
|---|---|---|
| **Windows 10/11** | ✅ Fully supported | MSVC, MinGW. Hybrid GPU systems handled via `NvOptimusEnablement`. |
| **Linux** | ✅ Fully supported | GCC, Clang. Requires NVIDIA proprietary driver. X11 or Wayland via XWayland. |
| **macOS** | ❌ Not supported | NVIDIA dropped CUDA support on macOS after CUDA 10.2 / macOS 10.15 (2019). Will not function on any modern Mac. |

### GPU Compatibility

| Architecture | Compute Capability | Example GPUs |
|---|---|---|
| Maxwell | 5.x | GTX 9xx series |
| Pascal | 6.x | GTX 10xx series |
| Turing | 7.5 | RTX 20xx, GTX 16xx |
| Ampere | 8.x | RTX 30xx series |
| Ada Lovelace | 8.9 | RTX 40xx series |
| Hopper | 9.0 | H100, H200 |

---

## Installation

### Option 1: FetchContent (Recommended)

Add to your `CMakeLists.txt`:

```cmake
include(FetchContent)

FetchContent_Declare(
    raylib_cuda
    GIT_REPOSITORY https://github.com/0bVdnt/raylib_cuda.git
    GIT_TAG        v1.1.1
)
FetchContent_MakeAvailable(raylib_cuda)

target_link_libraries(your_app PRIVATE raylib_cuda)
```

This pulls in both `raylib_cuda` and `raylib` — no need to fetch raylib separately.

### Option 2: Git Submodule

```bash
git submodule add https://github.com/0bVdnt/raylib_cuda.git external/raylib_cuda
```

```cmake
add_subdirectory(external/raylib_cuda)
target_link_libraries(your_app PRIVATE raylib_cuda)
```

### Option 3: Build from Source

```bash
git clone https://github.com/0bVdnt/raylib_cuda.git
cd raylib_cuda

# Configure (raylib will be fetched automatically)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release

# Install (optional)
cmake --install build --prefix /usr/local
```

#### Using a Local raylib

If you already have raylib source checked out:

```bash
cmake -B build -DRAYLIB_DIR=/path/to/raylib/source
cmake --build build
```

---

## Quick Start

### Minimal Example

```cuda
// main.cu — Must be a CUDA source file (.cu)
#include "raylib_cuda.h"
#include "raylib_cuda_kernel.cuh"

__global__ void fillGradient(cudaSurfaceObject_t surface,
                             int width, int height, float time)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float r = (float)x / width;
    float g = (float)y / height;
    float b = (sinf(time) + 1.0f) * 0.5f;

    rlcWritePixelF(surface, x, y, r, g, b, 1.0f);
}

int main(void)
{
    const int width = 800, height = 600;

    // 1. Create window (standard raylib — configure flags beforehand if needed)
    SetConfigFlags(FLAG_VSYNC_HINT);
    InitWindow(width, height, "raylib_cuda — Gradient");

    // 2. Initialize CUDA interop (must be AFTER InitWindow)
    if (!RLC_InitCUDA()) {
        printf("CUDA init failed: %s\n", RLC_ErrorString(RLC_GetLastError()));
        CloseWindow();
        return 1;
    }

    // 3. Create a surface (texture + CUDA registration)
    RLC_Surface surface = RLC_CreateSurface(width, height);
    if (!RLC_IsValid(&surface)) {
        printf("Surface creation failed\n");
        RLC_Close();
        return 1;
    }

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    float time = 0.0f;

    while (!WindowShouldClose())
    {
        time += GetFrameTime();

        // 4. Map for CUDA access
        cudaSurfaceObject_t surf = (cudaSurfaceObject_t)RLC_BeginAccess(&surface);
        if (surf != 0) {
            // 5. Launch kernel
            fillGradient<<<grid, block>>>(surf, width, height, time);
        }
        // 6. Unmap (synchronizes GPU, returns texture to raylib)
        RLC_EndAccess(&surface);

        // 7. Draw with raylib
        BeginDrawing();
            ClearBackground(BLACK);
            DrawTexture(RLC_GetTexture(&surface), 0, 0, WHITE);
            DrawFPS(10, 10);
        EndDrawing();
    }

    // 8. Cleanup
    RLC_UnloadSurface(&surface);
    RLC_Close();  // closes CUDA + window
    return 0;
}
```

### CMakeLists.txt for Your Project

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_cuda_app LANGUAGES C CXX CUDA)

include(FetchContent)
FetchContent_Declare(
    raylib_cuda
    GIT_REPOSITORY https://github.com/0bVdnt/raylib_cuda.git
    GIT_TAG        v1.1.1
)
FetchContent_MakeAvailable(raylib_cuda)

add_executable(my_app main.cu)
target_link_libraries(my_app PRIVATE raylib_cuda)
set_target_properties(my_app PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED ON
)
```

---

## Surface Formats

| Format | Bytes/Pixel | CUDA Type | Channels | Use Cases |
|---|---|---|---|---|
| `RLC_FORMAT_RGBA8` | 4 | `uchar4` | 4 × 8-bit | General visualization, color rendering |
| `RLC_FORMAT_R32F` | 4 | `float` | 1 × 32-bit | Heightmaps, distance fields, single-channel simulations |
| `RLC_FORMAT_RGBA32F` | 16 | `float4` | 4 × 32-bit | HDR rendering, physics buffers, precision-sensitive data |

```c
// Default (RGBA8)
RLC_Surface vis = RLC_CreateSurface(800, 600);

// Single-channel float
RLC_Surface heightmap = RLC_CreateSurfaceEx(512, 512, RLC_FORMAT_R32F);

// HDR float4
RLC_Surface hdr = RLC_CreateSurfaceEx(800, 600, RLC_FORMAT_RGBA32F);
```

> **Important:** Use the correct kernel helpers for your surface format. `rlcWritePixel` / `rlcReadPixel` use a 4-byte-per-pixel stride and are intended for `RLC_FORMAT_RGBA8` only. Using them on an `RLC_FORMAT_RGBA32F` surface will produce incorrect results. There is no runtime format check on the device side.

---

## API Reference

### Initialization & Shutdown

---

#### `RLC_InitCUDA()`

```c
bool RLC_InitCUDA(void);
```

Initializes the CUDA context for OpenGL interoperability. **Must be called after `InitWindow()`** — the CUDA–OpenGL interop requires an active OpenGL context, which raylib creates during `InitWindow()`. Calling `RLC_InitCUDA()` separately also allows you to configure raylib flags first.

**Returns:** `true` on success, `false` if no compatible CUDA device is found.

```c
SetConfigFlags(FLAG_VSYNC_HINT | FLAG_MSAA_4X_HINT);
InitWindow(800, 600, "My App");

if (!RLC_InitCUDA()) {
    printf("Error: %s\n", RLC_ErrorString(RLC_GetLastError()));
    CloseWindow();
    return 1;
}
```

---

#### `RLC_CloseCUDA()`

```c
void RLC_CloseCUDA(void);
```

Releases CUDA resources. Call before `CloseWindow()` when using the separate initialization pattern.

---

#### `RLC_Close()`

```c
void RLC_Close(void);
```

Convenience function that calls both `RLC_CloseCUDA()` and `CloseWindow()`.

---

#### `RLC_Init()` *(deprecated)*

```c
bool RLC_Init(int width, int height, const char *title);
```

Initializes window and CUDA together. **Deprecated** — use `InitWindow()` + `RLC_InitCUDA()` instead for more control over window configuration. Will be removed in v2.0.

---

### Surface Management

---

#### `RLC_CreateSurface()`

```c
RLC_Surface RLC_CreateSurface(int width, int height);
```

Creates an RGBA8 surface (4 bytes per pixel, 8 bits per channel).

**Returns:** `RLC_Surface` struct. Check validity with `RLC_IsValid()`.

---

#### `RLC_CreateSurfaceEx()`

```c
RLC_Surface RLC_CreateSurfaceEx(int width, int height, RLC_Format format);
```

Creates a surface with the specified pixel format. See [Surface Formats](#surface-formats) for available formats and their CUDA types.

---

#### `RLC_ResizeSurface()`

```c
bool RLC_ResizeSurface(RLC_Surface *surface, int newWidth, int newHeight);
```

Resizes an existing surface, preserving its format. **Surface must not be mapped** — call `RLC_EndAccess()` first.

**Returns:** `true` on success, `false` on failure.

---

#### `RLC_UnloadSurface()`

```c
void RLC_UnloadSurface(RLC_Surface *surface);
```

Frees all resources associated with a surface. Safe to call with `NULL` or already-freed surfaces. Automatically unmaps if still mapped.

---

#### `RLC_IsValid()`

```c
bool RLC_IsValid(const RLC_Surface *surface);
```

Checks if a surface is properly initialized and ready to use.

```c
RLC_Surface surf = RLC_CreateSurface(800, 600);
if (!RLC_IsValid(&surf)) {
    RLC_Error err = RLC_GetLastError();
    printf("Failed: %s\n", RLC_ErrorString(err));
}
```

---

#### `RLC_GetBytesPerPixel()`

```c
int RLC_GetBytesPerPixel(RLC_Format format);
```

Returns the number of bytes per pixel for a given format (4, 4, or 16).

---

### Execution Pipeline

---

#### `RLC_BeginAccess()`

```c
unsigned long long RLC_BeginAccess(RLC_Surface *surface);
```

Maps the surface for CUDA kernel access.

**Returns:** `cudaSurfaceObject_t` cast to `unsigned long long`. Returns `0` on failure — check `RLC_GetLastError()` for details.

**Important:** You **must** call `RLC_EndAccess()` or `RLC_EndAccessAsync()` before drawing the texture with raylib.

---

#### `RLC_EndAccess()`

```c
void RLC_EndAccess(RLC_Surface *surface);
```

Unmaps the surface and synchronizes the GPU. **Blocking call** — waits for all GPU work to complete. Safe to call even if `RLC_BeginAccess()` failed. Idempotent.

---

#### `RLC_EndAccessAsync()`

```c
void RLC_EndAccessAsync(RLC_Surface *surface);
```

Unmaps the surface **without** waiting for GPU completion. You must call `RLC_Sync()` or use CUDA events/streams to ensure work is complete before drawing.

---

#### `RLC_Sync()`

```c
void RLC_Sync(void);
```

Blocks until all GPU work completes. Use after `RLC_EndAccessAsync()` before drawing.

---

#### `RLC_IsMapped()`

```c
bool RLC_IsMapped(const RLC_Surface *surface);
```

Returns `true` if the surface is currently mapped for CUDA access.

---

### Usage Patterns

**Standard pattern:**

```cuda
cudaSurfaceObject_t surf = (cudaSurfaceObject_t)RLC_BeginAccess(&surface);
if (surf != 0) {
    myKernel<<<grid, block>>>(surf, width, height);
}
RLC_EndAccess(&surface);

// Now safe to draw
DrawTexture(RLC_GetTexture(&surface), 0, 0, WHITE);
```

**Double-buffering pattern** (for simulations that read from one surface and write to another):

```cuda
cudaSurfaceObject_t src = (cudaSurfaceObject_t)RLC_BeginAccess(current);
cudaSurfaceObject_t dst = (cudaSurfaceObject_t)RLC_BeginAccess(next);

if (src != 0 && dst != 0) {
    simulationStep<<<grid, block>>>(src, dst, width, height);
}

RLC_EndAccess(current);
RLC_EndAccess(next);

// Swap
RLC_Surface *temp = current;
current = next;
next = temp;

// Draw the latest result
DrawTexture(RLC_GetTexture(current), 0, 0, WHITE);
```

**Async pattern** (overlap CPU work with GPU computation):

```cuda
cudaSurfaceObject_t surf = (cudaSurfaceObject_t)RLC_BeginAccess(&surface);
if (surf != 0) {
    myKernel<<<grid, block>>>(surf, width, height);
}
RLC_EndAccessAsync(&surface);  // returns immediately

// Do CPU work here while GPU is still running...
doExpensiveCPUWork();

// Sync before drawing
RLC_Sync();
DrawTexture(RLC_GetTexture(&surface), 0, 0, WHITE);
```

---

### Utility Functions

---

#### `RLC_GetTexture()`

```c
Texture2D RLC_GetTexture(const RLC_Surface *surface);
```

Returns the raylib `Texture2D` for drawing. Returns a zeroed struct if `surface` is `NULL`.

---

#### `RLC_GetWidth()` / `RLC_GetHeight()`

```c
int RLC_GetWidth(const RLC_Surface *surface);
int RLC_GetHeight(const RLC_Surface *surface);
```

Returns surface dimensions in pixels. Returns `0` if `surface` is `NULL`.

---

#### `RLC_GetFormat()`

```c
RLC_Format RLC_GetFormat(const RLC_Surface *surface);
```

Returns the pixel format of the surface.

---

#### `RLC_GetVersion()`

```c
void RLC_GetVersion(int *major, int *minor, int *patch);
```

Returns the library version numbers. Any parameter may be `NULL`.

---

### Error Handling

---

#### `RLC_GetLastError()`

```c
RLC_Error RLC_GetLastError(void);
```

Returns and **clears** the last error code.

> **Note:** The error state is global and cleared on read. Always check errors immediately after the call that might fail. An unchecked error will be silently lost. The error state is not thread-safe — do not call the library from multiple threads simultaneously.

---

#### `RLC_ErrorString()`

```c
const char* RLC_ErrorString(RLC_Error error);
```

Returns a human-readable error description.

---

#### Error Codes

| Code | Description |
|---|---|
| `RLC_OK` | No error |
| `RLC_ERROR_NO_CUDA_DEVICE` | No CUDA-capable GPU found |
| `RLC_ERROR_WRONG_GPU` | Intel integrated GPU detected — need discrete NVIDIA GPU |
| `RLC_ERROR_INIT_FAILED` | CUDA initialization failed (e.g., window not created yet) |
| `RLC_ERROR_INVALID_ARGUMENT` | Invalid function argument (e.g., zero or negative dimensions) |
| `RLC_ERROR_UNSUPPORTED_FORMAT` | Unsupported pixel format |
| `RLC_ERROR_REGISTER_FAILED` | Failed to register texture with CUDA |
| `RLC_ERROR_MAP_FAILED` | Failed to map surface for CUDA access |
| `RLC_ERROR_NOT_MAPPED` | Operation requires a mapped surface |
| `RLC_ERROR_ALREADY_MAPPED` | Surface is already mapped |
| `RLC_ERROR_NULL_SURFACE` | NULL surface pointer provided |

**Error handling pattern:**

```c
unsigned long long handle = RLC_BeginAccess(&surface);
if (handle == 0) {
    RLC_Error err = RLC_GetLastError();
    printf("BeginAccess failed: %s (code %d)\n", RLC_ErrorString(err), err);
    // handle error...
}
```

---

## Kernel Helper Functions

Include `raylib_cuda_kernel.cuh` in your `.cu` files for convenient device-side pixel operations.

### RGBA8 Format

```cuda
// Write pixel with byte colors (0–255)
__device__ void rlcWritePixel(cudaSurfaceObject_t surf, int x, int y,
                              unsigned char r, unsigned char g,
                              unsigned char b, unsigned char a = 255);

// Write pixel with float colors (0.0–1.0, auto-clamped)
__device__ void rlcWritePixelF(cudaSurfaceObject_t surf, int x, int y,
                               float r, float g, float b, float a = 1.0f);

// Read pixel
__device__ uchar4 rlcReadPixel(cudaSurfaceObject_t surf, int x, int y);
```

### R32F Format (Single Float)

```cuda
__device__ void rlcWriteFloat(cudaSurfaceObject_t surf, int x, int y, float value);
__device__ float rlcReadFloat(cudaSurfaceObject_t surf, int x, int y);
```

### RGBA32F Format (Four Floats)

```cuda
__device__ void rlcWriteFloat4(cudaSurfaceObject_t surf, int x, int y,
                               float r, float g, float b, float a = 1.0f);
__device__ float4 rlcReadFloat4(cudaSurfaceObject_t surf, int x, int y);
```

### Bounds-Checked Variants

All write functions have `*Safe` variants that silently skip out-of-bounds writes:

```cuda
__device__ bool rlcInBounds(int x, int y, int width, int height);

__device__ void rlcWritePixelSafe(cudaSurfaceObject_t surf, int x, int y,
                                  int width, int height,
                                  unsigned char r, unsigned char g,
                                  unsigned char b, unsigned char a = 255);

__device__ void rlcWritePixelFSafe(cudaSurfaceObject_t surf, int x, int y,
                                   int width, int height,
                                   float r, float g, float b, float a = 1.0f);

__device__ void rlcWriteFloatSafe(cudaSurfaceObject_t surf, int x, int y,
                                  int width, int height, float value);

__device__ void rlcWriteFloat4Safe(cudaSurfaceObject_t surf, int x, int y,
                                   int width, int height,
                                   float r, float g, float b, float a = 1.0f);
```

---

## Examples

### Game of Life

The included [`examples/game_of_life.cu`](examples/game_of_life.cu) is a complete Conway's Game of Life running entirely on the GPU with real-time visualization.

**Features demonstrated:**
- Double-buffered simulation (read from one surface, write to another)
- CUDA kernel pixel read/write with toroidal wrapping
- Surface initialization with GPU-side random data
- Interactive controls
- Proper error handling at every stage

**Controls:**

| Key | Action |
|---|---|
| `Space` | Pause / resume simulation |
| `R` | Randomize grid |
| `Esc` | Quit |

**Build and run:**

```bash
cmake -B build -DRLC_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/examples/game_of_life
```

---

## Build Configuration

### CMake Options

| Option | Default | Description |
|---|---|---|
| `RLC_BUILD_EXAMPLES` | `ON` | Build example programs |
| `RLC_BUILD_TESTS` | `OFF` | Build test programs |
| `RAYLIB_DIR` | `""` | Path to local raylib source (auto-fetches if empty) |
| `RAYLIB_VERSION` | `"5.5"` | raylib version to fetch from GitHub |

### Specifying CUDA Architectures

```bash
# Build for specific architectures
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="75;86;89"

# Build for native GPU only (CMake 3.24+, auto-detected by default)
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
```

Common architecture values:

| Value | GPU Family |
|---|---|
| `52` | Maxwell (GTX 9xx) |
| `61` | Pascal (GTX 10xx) |
| `75` | Turing (RTX 20xx, GTX 16xx) |
| `86` | Ampere (RTX 30xx) |
| `89` | Ada Lovelace (RTX 40xx) |
| `90` | Hopper (H100, datacenter) |

### Build Types

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release  # Optimized
cmake -B build -DCMAKE_BUILD_TYPE=Debug    # With debug symbols
```

### Advanced: Skip OpenGL Synchronization

By default, `RLC_BeginAccess()` calls `glFinish()` to ensure all pending OpenGL operations complete before CUDA takes ownership. This is a full pipeline stall but guarantees safety.

If you are not mixing raw OpenGL draw calls with CUDA access between frames (the common case with this library), you can skip this for better performance:

```cmake
target_compile_definitions(raylib_cuda PRIVATE RLC_SKIP_GL_SYNC)
```

> **Warning:** May cause visual artifacts if OpenGL and CUDA operations overlap.

---

## Project Structure

```
raylib_cuda/
├── CMakeLists.txt                  # Main build configuration
├── LICENSE
├── README.md
├── include/
│   ├── raylib_cuda.h               # Public C API
│   └── raylib_cuda_kernel.cuh      # CUDA kernel helpers (device-side)
├── src/
│   ├── raylib_cuda.c               # C implementation (no GPU/GL headers)
│   └── raylib_cuda_backend.cu      # CUDA/OpenGL interop backend
├── examples/
│   ├── CMakeLists.txt
│   └── game_of_life.cu             # Conway's Game of Life on the GPU
└── tests/                          # Test programs (if present)
```

---

## Platform-Specific Notes

### Windows

The library automatically requests the discrete NVIDIA GPU on hybrid systems via `NvOptimusEnablement` and `AmdPowerXpressRequestHighPerformance` exports. If still using Intel GPU:

1. Right-click executable → *Run with graphics processor* → *NVIDIA*
2. Or configure in *NVIDIA Control Panel* → *Manage 3D Settings* → *Program Settings* → add your app → set to *High-performance NVIDIA processor*

### Linux

For NVIDIA Optimus/PRIME systems:

```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./your_app

# Or use prime-run if available
prime-run ./your_app
```

---

## Troubleshooting

### "No CUDA device found"

1. Verify NVIDIA drivers are installed: `nvidia-smi`
2. Verify CUDA toolkit is installed: `nvcc --version`
3. Check that your GPU supports CUDA (GeForce, Quadro, Tesla, or RTX series)

### "Intel GPU detected" / "Wrong GPU"

Your application's OpenGL context is running on the Intel integrated GPU instead of the NVIDIA discrete GPU. CUDA kernels cannot access a texture owned by a different GPU. See [Platform-Specific Notes](#platform-specific-notes) for how to force the NVIDIA GPU.

### "Window must be initialized before calling RLC_InitCUDA()"

`RLC_InitCUDA()` was called before `InitWindow()`. The OpenGL context must exist first:

```c
// Wrong
RLC_InitCUDA();
InitWindow(800, 600, "App");

// Correct
InitWindow(800, 600, "App");
RLC_InitCUDA();
```

### "Failed to register texture with CUDA"

1. Update NVIDIA drivers to the latest version
2. Verify `InitWindow()` succeeded (`IsWindowReady()` returns true)
3. Check that the OpenGL context is valid

### "Cannot resize mapped surface"

Call `RLC_EndAccess()` before `RLC_ResizeSurface()`.

### Black screen / No output

1. Check that `RLC_BeginAccess()` returned a non-zero handle
2. Ensure your kernel has bounds checking: `if (x >= width || y >= height) return;`
3. Ensure `RLC_EndAccess()` is called **before** `BeginDrawing()` / `DrawTexture()`
4. Verify kernel grid/block dimensions cover the full texture
5. For debugging, add `cudaGetLastError()` and `cudaDeviceSynchronize()` after kernel launches

### Low performance

1. Minimize `RLC_BeginAccess()` / `RLC_EndAccess()` calls per frame — batch work into single map/unmap cycles
2. Use `RLC_EndAccessAsync()` + `RLC_Sync()` to overlap CPU work with GPU
3. Build in Release mode (`-DCMAKE_BUILD_TYPE=Release`)
4. Consider defining `RLC_SKIP_GL_SYNC` if not mixing GL and CUDA (see [Advanced: Skip OpenGL Synchronization](#advanced-skip-opengl-synchronization))

### Build errors about missing CUDA architectures

If CMake < 3.24, set architectures manually:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86
```

Use the value matching your GPU (see [Build Configuration](#build-configuration)).

### Linking errors on Windows

Ensure `opengl32` is being linked. The CMakeLists handles this automatically, but if integrating manually:

```cmake
target_link_libraries(my_app PRIVATE raylib_cuda opengl32)
```

---

## Version History

### v1.1.1 (Current)
- Fixed surface object handle management
- Improved error messages
- Documentation updates

### v1.1.0
- Added `RLC_InitCUDA()` for flexible window configuration
- Added `RLC_Format` enum with RGBA8, R32F, RGBA32F support
- Added `RLC_CreateSurfaceEx()`, `RLC_ResizeSurface()`, `RLC_EndAccessAsync()`
- Added kernel helper functions (`rlcWritePixel`, `rlcReadPixel`, bounds-checked variants)
- Deprecated `RLC_Init()`

### v1.0.0
- Initial release

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m "Add amazing feature"`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Code Style

- C99 for `.c` files, C++11 for `.cu` files
- 4-space indentation, Allman braces
- `RLC_PascalCase` for public API, `rlc_snake_case` for internal functions, `rlcCamelCase` for kernel helpers

---

## License

MIT License

---

## Acknowledgments

- **[raylib](https://github.com/raysan5/raylib)** — Simple and enjoyable game programming library by Ramon Santamaria
- **[NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)** — Parallel computing platform and API

---

<div align="center">

**[⬆ Back to Top](#raylib_cuda)**

Made with ❤️ for the raylib and CUDA communities

</div>
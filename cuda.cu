#include "cuda_kernels.h"
#include <stdio.h>
#include <chrono>
#include <iostream>

#define BLOCK_SIZE 256

// Persistent device pointers
static bool* d_grid = nullptr;
static bool* d_temp = nullptr;
static int d_width = 0;
static int d_height = 0;

__global__ void cudaStep(bool* grid, bool* currGrid, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) return;

    int x = i % width;
    int y = i / width;

    int idx = y * width + x;
    int neighbours = 0;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            if (dx == 0 && dy == 0) continue;
            int nx = (x + dx + width) % width;
            int ny = (y + dy + height) % height;
            neighbours += grid[ny * width + nx] ? 1 : 0;
        }
    }
    bool alive = grid[idx];
    currGrid[idx] = (alive && (neighbours == 2 || neighbours == 3)) || (!alive && neighbours == 3);
}

void init_cuda(bool* host_grid, int width, int height) {
    d_width = width;
    d_height = height;
    size_t size = width * height * sizeof(bool);
    
    cudaMalloc(&d_grid, size);
    cudaMalloc(&d_temp, size);
    cudaMemcpy(d_grid, host_grid, size, cudaMemcpyHostToDevice);
}

void step_cuda() {
    if (!d_grid || !d_temp) return;  // Ensure initialized
    
    int numBlocks = (d_width * d_height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaStep<<<numBlocks, BLOCK_SIZE>>>(d_grid, d_temp, d_width, d_height);
    cudaDeviceSynchronize();
    
    
    bool* temp = d_grid;
    d_grid = d_temp;
    d_temp = temp;
}

void get_grid_cuda(bool* host_grid, int width, int height) {
    if (!d_grid) return;
    size_t size = width * height * sizeof(bool);
    cudaMemcpy(host_grid, d_grid, size, cudaMemcpyDeviceToHost);
}

void cleanup_cuda() {
    if (d_grid) cudaFree(d_grid);
    if (d_temp) cudaFree(d_temp);
    d_grid = nullptr;
    d_temp = nullptr;
}
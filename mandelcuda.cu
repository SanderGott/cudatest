#include "mandel.h"
#include <iostream>

#define BLOCK_SIZE 256

__global__ void cudaPixel(RGB *cudaScreen, int width, int height, double cx, double cy, double zoom)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height)
        return;

    int Sx = i % width;
    int Sy = i / width;

    double init_x_min = -2.0;
    double init_x_max = 1.0;
    double init_y_min = -1.0;
    double init_y_max = 1.0;

    double x_half = (init_x_max - init_x_min) / (2.0 * zoom);
    double y_half = (init_y_max - init_y_min) / (2.0 * zoom);

    double x_min = cx - x_half;
    double x_max = cx + x_half;
    double y_min = cy - y_half;
    double y_max = cy + y_half;

    float x0 = x_min + (Sx / (width - 1.0)) * (x_max - x_min);
    float y0 = y_min + (Sy / (height - 1.0)) * (y_max - y_min);

    float x = 0.0f;
    float y = 0.0f;
    int iteration = 0;
    while (x * x + y * y <= 4.0f && iteration < MAX_ITERS)
    {
        float x2 = x * x - y * y + x0;
        float y2 = 2.0f * x * y + y0;
        x = x2;
        y = y2;
        iteration++;
    }

    // cudaScreen[Sy * width + Sx] = iteration;
    int idx = Sy * width + Sx;

    if (iteration >= MAX_ITERS)
    {
        cudaScreen[idx].r = 0;
        cudaScreen[idx].g = 0;
        cudaScreen[idx].b = 0;
    }
    else
    {
        float t = (float)iteration / (float)MAX_ITERS;
        unsigned char r = (unsigned char)(9.0f * (1 - t) * t * t * t * 255.0f);
        unsigned char g = (unsigned char)(15.0f * (1 - t) * (1 - t) * t * t * 255.0f);
        unsigned char b = (unsigned char)(8.5f * (1 - t) * (1 - t) * (1 - t) * t * 255.0f);
        cudaScreen[idx].r = r;
        cudaScreen[idx].g = g;
        cudaScreen[idx].b = b;
    }
}

void computeMandelCuda(RGB *screen, int width, int height, double cx, double cy, double zoom)
{

    RGB *cudaScreen;
    cudaMalloc(&cudaScreen, width * height * sizeof(RGB));

    int numBlocks = (width * height + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaPixel<<<numBlocks, BLOCK_SIZE>>>(cudaScreen, width, height, cx, cy, zoom);

    cudaDeviceSynchronize();

    cudaMemcpy(screen, cudaScreen, width * height * sizeof(RGB), cudaMemcpyDeviceToHost);

    cudaFree(cudaScreen);

    // std::cout << "here!" << std::endl;
    // std::cout << screen[2034] << std::endl;
}
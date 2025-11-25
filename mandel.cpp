#include "mandel.h"

int mandel_iterations(float x0, float y0, int max_iter)
{
    float x = 0.0f;
    float y = 0.0f;
    int iteration = 0;
    while (x * x + y * y <= 4.0f && iteration < max_iter)
    {
        float x2 = x * x - y * y + x0;
        float y2 = 2.0f * x * y + y0;
        x = x2;
        y = y2;
        iteration++;
    }
    return iteration;
}

void mandelstuff(RGB *screen, int width, int height, double cx, double cy, double zoom)
{
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

    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < height; ++j)
        {
            double x0 = x_min + ((double)i / (double)(width - 1)) * (x_max - x_min);
            double y0 = y_min + ((double)j / (double)(height - 1)) * (y_max - y_min);

            int iters = mandel_iterations((float)x0, (float)y0, MAX_ITERS);

            int idx = j * width + i;

            if (iters >= MAX_ITERS)
            {
                screen[idx].r = 0;
                screen[idx].g = 0;
                screen[idx].b = 0;
            }
            else
            {
                float t = (float)iters / (float)MAX_ITERS;
                unsigned char r = (unsigned char)(9.0f * (1 - t) * t * t * t * 255.0f);
                unsigned char g = (unsigned char)(15.0f * (1 - t) * (1 - t) * t * t * 255.0f);
                unsigned char b = (unsigned char)(8.5f * (1 - t) * (1 - t) * (1 - t) * t * 255.0f);
                screen[idx].r = r;
                screen[idx].g = g;
                screen[idx].b = b;
            }
        }
    }
}
#pragma once

struct RGB
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

void mandelstuff(RGB *screen, int width, int height, double cx, double cy, double zoom);
void computeMandelCuda(RGB *screen, int width, int height, double cx, double cy, double zoom);

#define MAX_ITERS 300

#pragma once

void init_cuda(bool *host_grid, int width, int height);

void step_cuda();

void get_grid_cuda(bool *host_grid, int width, int height);

void cleanup_cuda();
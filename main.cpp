#include <iostream>
#include <SDL2/SDL.h>
#include <omp.h>
#include "mandel.h"
#ifdef cuda
#include "cuda_kernels.h"
#endif

/*
litt over 200 seq
litt over 400 omp

*/

void update_loop(bool *grid, bool *newGrid, int width, int height)
{
    // #pragma omp parallel for
    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int pos = y * width + height;

            int idx = y * width + x;
            int neighbours = 0;
            for (int dx = -1; dx <= 1; ++dx)
            {
                for (int dy = -1; dy <= 1; ++dy)
                {
                    if (dx == 0 && dy == 0)
                        continue;
                    int nx = (x + dx + width) % width;
                    int ny = (y + dy + height) % height;
                    neighbours += grid[ny * width + nx] ? 1 : 0;
                }
            }
            bool alive = grid[idx];
            newGrid[idx] = (alive && (neighbours == 2 || neighbours == 3)) || (!alive && neighbours == 3);
        }
    }
}

int main()
{
    std::cout << "Starting to execute!" << std::endl;

    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
        return 1;
    }
    int width = 1900;
    int height = 1200;

    SDL_Window *window = SDL_CreateWindow("Game of Life", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_SHOWN);
    if (!window)
    {
        std::cerr << "SDL_CreateWindow Error: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer)
    {
        std::cerr << "SDL_CreateRenderer Error: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Create texture for pixel data (RGBA format for simplicity)
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, width, height);
    if (!texture)
    {
        std::cerr << "SDL_CreateTexture Error: " << SDL_GetError() << std::endl;
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    const int TARGET_FPS = 60;
    const Uint32 TARGET_FRAME_TIME = 1000 / TARGET_FPS; // Time per frame in milliseconds

    bool *grid = new bool[width * height];
    bool *tempGrid = new bool[width * height];
    bool *cudaTempGrid = new bool[width * height];

    RGB *screen = new RGB[width * height];

    for (int i = 0; i < width * height; i++)
    {
        grid[i] = false;
    }

    Uint32 *pixels = new Uint32[width * height];

    for (int i = 0; i < width * height; ++i)
    {
        pixels[i] = 0x000000FF; // Black
    }

    auto setPixel = [&](int x, int y, bool alive)
    {
        if (x >= 0 && x < width && y >= 0 && y < height)
        {
            pixels[y * width + x] = alive ? 0xFFFFFFFF : 0x000000FF;
        }
    };

    auto rgb_to_pixel = [](const RGB &c) -> Uint32
    {
        return ((Uint32)c.r << 24) | ((Uint32)c.g << 16) | ((Uint32)c.b << 8) | 0xFFu;
    };

    setPixel(0, 0, true);

    for (int i = 0; i < width; i++)
    {
        setPixel(i, 100, true);
        grid[i] = true;
    }

#ifdef cuda
    init_cuda(grid, width, height);
#endif

    Uint32 lastFpsTime = SDL_GetTicks();
    int frameCount = 0;

    double cx = 0;
    double cy = 0;
    double zoom = 1;

    SDL_Event event;
    bool running = true;
    while (running)
    {
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
            {
                running = false;
            }
            else if (event.type == SDL_KEYDOWN)
            {
                double pan_step = 0.1 / zoom; // pan amount scales with zoom
                switch (event.key.keysym.sym)
                {
                case SDLK_LEFT:
                    cx -= pan_step;
                    break;
                case SDLK_RIGHT:
                    cx += pan_step;
                    break;
                case SDLK_UP:
                    cy -= pan_step;
                    break;
                case SDLK_DOWN:
                    cy += pan_step;
                    break;
                case SDLK_KP_PLUS:
                case SDLK_EQUALS:
                    zoom *= 1.25;
                    break;
                case SDLK_KP_MINUS:
                case SDLK_MINUS:
                    zoom /= 1.25;
                    if (zoom < 1e-12)
                        zoom = 1e-12;
                    break;
                case SDLK_r:
                    cx = 0.0;
                    cy = 0.0;
                    zoom = 1.0;
                    break;
                default:
                    break;
                }
                std::cout << "view: cx=" << cx << " cy=" << cy << " zoom=" << zoom << std::endl;
            }
        }
#ifdef cuda
        step_cuda();
        get_grid_cuda(tempGrid, width, height);
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                int pos = y * width + height;
                if (tempGrid[pos])
                {
                    setPixel(x, y, true);
                }
                else
                {
                    setPixel(x, y, false);
                }
            }
        }
#elif cpu
        update_loop(grid, tempGrid, width, height);
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                int pos = y * width + height;
                if (tempGrid[pos])
                {
                    setPixel(x, y, true);
                }
                else
                {
                    setPixel(x, y, false);
                }
            }
        }

#elif mandel
        mandelstuff(screen, width, height, cx, cy, zoom);

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int idx = y * width + x;
                RGB col = screen[idx];
                pixels[idx] = rgb_to_pixel(col);
            }
        }

#elif mandelcuda
        computeMandelCuda(screen, width, height, cx, cy, zoom);

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int idx = y * width + x;
                RGB col = screen[idx];
                pixels[idx] = rgb_to_pixel(col);
            }
        }

#endif
        // update_loop(grid, tempGrid, width, height);

        // for (int i = 0; i < width*height; i++)
        // {
        //     if (tempGrid[i] != cudaTempGrid[i])
        //     {
        //         std::cout << "Versions not matching!!!" << std::endl;
        //         running = false;
        //         break;
        //     }
        // }

#ifdef cpu
        memcpy(grid, tempGrid, width * height * sizeof(bool));
#endif

        void *texturePixels;
        int pitch;
        SDL_LockTexture(texture, NULL, &texturePixels, &pitch);
        memcpy(texturePixels, pixels, width * height * sizeof(Uint32));
        SDL_UnlockTexture(texture);

        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);

        frameCount++;
        Uint32 currentTime = SDL_GetTicks();
        if (currentTime - lastFpsTime >= 1000)
        {
            float fps = frameCount / ((currentTime - lastFpsTime) / 1000.0f);
            std::cout << "FPS: " << fps << std::endl;
            frameCount = 0;
            lastFpsTime = currentTime;
        }

        Uint32 frameStart = SDL_GetTicks(); // Record start time of the frame

        Uint32 frameEnd = SDL_GetTicks(); // Record end time
        Uint32 frameTime = frameEnd - frameStart;

        // //if (frameTime < TARGET_FRAME_TIME) {
        // //    SDL_Delay(TARGET_FRAME_TIME - frameTime);  // Delay to cap FPS
        // //}
    }

    delete[] pixels;
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

#include <math.h> //for sqrt function
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

struct L
{
    long long int a;
    long long int b;
};
struct L match(int k, long long int i, long long int j, int x)
{
    struct L res = {0, 0};
    switch (k)
    {
    case 0:
        res.a = i - x;
        res.b = j - x;
        break;
    case 1:
        res.a = i - x;
        res.b = j;
        break;
    case 2:
        res.a = i - x;
        res.b = j + x;
        break;
    case 3:
        res.a = i;
        res.b = j - x;
        break;
    case 4:
        res.a = i;
        res.b = j;
        break;
    case 5:
        res.a = i;
        res.b = j + x;
        break;
    case 6:
        res.a = i + x;
        res.b = j - x;
        break;
    case 7:
        res.a = i + x;
        res.b = j;
        break;
    case 8:
        res.a = i + x;
        res.b = j + x;
        break;
    }
    return res;
}

void compute1(float **A, int **B, long long int N, int x)
{
    float V[9];
    for (long long int i = x; i < N - x; i++)
    {
        for (long long int j = x; j < N - x; j++)
        {
            for (int z = 0; z < 9; z++)
            {
                struct L tmp = match(z, i, j, x);
                V[z] = sqrt(A[tmp.a][tmp.b]);
            }
            int m = 4;
            for (int k = 0; k < 9; k++)
            {
                if (V[k] < V[m])
                    m = k;
            }
            B[i][j] = m;
        }
    }
}
struct L fnc0(int **B, long long int i, long long int j, int x)
{
    if (B[i][j] == 4)
    {
        struct L res = {i, j};
        return res;
    }
    else
    {
        struct L tmp = match(B[i][j], i, j, x);
        return fnc0(B, tmp.a, tmp.b, x);
    }
}
struct L compute2(int **B, long long int N, int x)
{
    int c1 = (rand() % N);
    int c2 = (rand() % N);
    struct L tmp = fnc0(B, c1, c2, x);
    return tmp;
}

int main(int argc, char *argv[])
{
    srand(time(0));
    float **A;
    int **B;
    int **B2;
    int **B3;
    long long int N = atoll(argv[1]);
    int argumentLoop = atoi(argv[2]);

    // Allocate row pointer arrays
    A = (float **)malloc(N * sizeof(float *));
    B = (int **)malloc(N * sizeof(int *));
    B2 = (int **)malloc(N * sizeof(int *));
    B3 = (int **)malloc(N * sizeof(int *));
    if (!A || !B || !B2 || !B3)
    {
        fprintf(stderr, "Failed to allocate row pointers\n");
        return 1;
    }

    // Allocate each row
    for (long long int i = 0; i < N; i++)
    {
        A[i] = (float *)malloc(N * sizeof(float));
        B[i] = (int *)malloc(N * sizeof(int));
        B2[i] = (int *)malloc(N * sizeof(int));
        B3[i] = (int *)malloc(N * sizeof(int));

        if (!A[i] || !B[i] || !B2[i] || !B3[i])
        {
            fprintf(stderr, "Failed to allocate matrix row %lld\n", i);
            return 1;
        }
    }

    // Initialize A with random data, B/B2/B3 with 4
    for (long long int i = 0; i < N; i++)
    {
        for (long long int j = 0; j < N; j++)
        {
            A[i][j] = (float)rand() / (float)RAND_MAX * 100.0f; // some positive float
            B[i][j] = 4;
            B2[i][j] = 4;
            B3[i][j] = 4;
        }
    }

    // Allocation for matrix A, B, B2, B3
    //  .....
    // Here, consider A, B, B2 and B3 as already initialised
    // A is initialised by reading files. All values of B, B2 and B3 are initialized to 4
    compute1(A, B, N, 1);
    compute1(A, B2, N, 2);
    compute1(A, B3, N, 3);

    struct L* res = new L[argumentLoop];
    struct L* res2= new L[argumentLoop];
    struct L* res3= new L[argumentLoop];

    for (int i = 0; i < argumentLoop; i++)
    {
        struct L r = compute2(B, N, 1);
        struct L r2 = compute2(B2, N, 2);
        struct L r3 = compute2(B3, N, 3);
        
        res[i]  = r;
        res2[i] = r2;
        res3[i] = r3;
    }

    for (int i = 0; i < (argumentLoop < 10 ? argumentLoop : 10); i++) {
        printf("res[%d]  = (%lld, %lld)\n", i, res[i].a,  res[i].b);
        printf("res2[%d] = (%lld, %lld)\n", i, res2[i].a, res2[i].b);
        printf("res3[%d] = (%lld, %lld)\n", i, res3[i].a, res3[i].b);
    }

    for (long long int i = 0; i < N; i++)
    {
        free(A[i]);
        free(B[i]);
        free(B2[i]);
        free(B3[i]);
    }
    free(A);
    free(B);
    free(B2);
    free(B3);

    // Free result arrays
    delete[] (res);
    delete[] (res2);
    delete[] (res3);

    return 0;
}

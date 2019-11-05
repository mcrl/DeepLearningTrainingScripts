#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <string>

#include "utils.h"

vector<string> split(string s, char delim) {
    vector<string> res;
    string tmp = "";
    for (int i = 0; i < s.size(); i++) {
        if( s[i] == delim ) {
                res.push_back(tmp);
            tmp.clear();
        }
        else tmp += s[i];
    }
    res.push_back(tmp);
    return res;
}

void readEntireFile (const char *fname, char** str) {
    FILE *f = fopen(fname, "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if ( fsize == 0 ) {
        fprintf(stderr, "EMPTY FILE\n");
        exit(0);
    }

    *str = (char*) malloc(fsize + 1);
    fread(*str, 1, fsize, f);
    fclose(f);

    (*str)[fsize] = 0;
}

int argmax(float *arr, int sz){
    float mx = -1e18; int mxx = -1;
    for (int i = 0; i < sz; i++) {
        if( arr[i] > mx ) mx = arr[i], mxx = i;
    }
    return mxx;
}

static float gauss()
{
    float x = (float)rand() / RAND_MAX;
    float y = (float)rand() / RAND_MAX;
    float z = sqrt(-2 * log(x)) * cos(2 * acos(-1) * y);
    return z;
}

static float gauss(float stdev) {
    return gauss() * stdev;
}

float char_to_float(char c) {
    unsigned char uc = (unsigned char) c;
    return (float) (uc);
}


void initRand (float *d_mem, int sz, float stdev, int ndev) {

    CUDA_CALL( cudaSetDevice(ndev) );

    float *tmp = (float*) malloc( sz * sizeof(float) );
    for (int i = 0; i < sz; i++) tmp[i] = gauss(stdev);
    CUDA_CALL( cudaMemcpy(d_mem, tmp, sz * sizeof(float), cudaMemcpyHostToDevice) );
    cudaDeviceSynchronize();
    free(tmp);
}

void initRandUniform (float *d_mem, int sz, float mx, int ndev) {

    CUDA_CALL( cudaSetDevice(ndev) );

    float *tmp = (float*) malloc( sz * sizeof(float) );
    for (int i = 0; i < sz; i++) {
        float t = (float) 2 * rand() / RAND_MAX;
        t -= 1.0;
        t *= mx;
        tmp[i] = t;
    }
    CUDA_CALL( cudaMemcpy(d_mem, tmp, sz * sizeof(float), cudaMemcpyHostToDevice) );
    cudaDeviceSynchronize();
    free(tmp);
}

void debugDevice (float *d_mem, int sz, int ndev) {
    CUDA_CALL( cudaSetDevice(ndev) );
    cudaDeviceSynchronize();
    float *tmp = (float*) malloc( sz * sizeof(float) );
    for (int i = 0; i < sz; i++) tmp[i] = i;
    CUDA_CALL( cudaMemcpy(tmp, d_mem, sz * sizeof(float), cudaMemcpyDeviceToHost) );
    cudaDeviceSynchronize();
    printf("process=%d ndev=%d\n", mpi_world_rank, ndev);
    printf("------------------------------------------------------\n");
    for (int i = 0; i < sz; i++) printf("%.4f ",tmp[i]);
    printf("\n------------------------------------------------------\n");
    free(tmp);
}

void debugDeviceInt (int *d_mem, int sz, int ndev) {
    printf("[debugDeviceInt] %x %d %d\n", d_mem, sz, ndev);
    CUDA_CALL( cudaSetDevice(ndev) );

    cudaDeviceSynchronize();
    int *tmp = (int*) malloc( sz * sizeof(int) );
    for (int i = 0; i < sz; i++) tmp[i] = i;
    CUDA_CALL( cudaMemcpy(tmp, d_mem, sz * sizeof(int), cudaMemcpyDeviceToHost) );
    cudaDeviceSynchronize();
    printf("process=%d ndev=%d\n", mpi_world_rank, ndev);
    printf("------------------------------------------------------\n");
    for (int i = 0; i < sz; i++) printf("%d ",tmp[i]);
    printf("\n------------------------------------------------------\n");
    free(tmp);
}


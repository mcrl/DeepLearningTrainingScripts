#ifndef _UTILS_H_
#define _UTILS_H_

#include <cstdio>
#include <cstdlib>
#include <math.h>

#include <vector>
#include <string>
#include <mpi.h>

#include "configs.h"

using namespace std;
using std::vector;
using std::string;

vector<string> split(string s, char delim);

void readEntireFile (const char *fname, char** str);

int argmax(float *arr, int sz);

static float gauss();
static float gauss(float stdev);
float char_to_float(char c);
void initRand (float *d_mem, int sz, float stdev, int ndev);
void initRandUniform (float *d_mem, int sz, float mx, int ndev);

void debugDevice (float *d_mem, int sz, int ndev);
void debugDeviceInt (int *d_mem, int sz, int ndev);
#endif // _UTILS_H_

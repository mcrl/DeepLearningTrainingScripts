#ifndef _DATA_H_
#define _DATA_H_

#include "configs.h"
#include "utils.h"

#include <math.h>
#include <string>
#include <iostream>
#include <vector>

#define TEST_DATA_SIZE 500
#define TRAIN_DATA_SIZE 1000
#define EPOCHS 2
#define BATCH_SIZE 10
#define DATA_FILE_LINES 45840617

using namespace std;

extern vector<string> featureList[50];
int getIdx(vector<string>& sv, string &s);

class Data{
public:
    int res;
    int dense[13];
    string sparse[26];
    int processed_sparse[26];

    Data ();

    Data (string s, int processed);

    ~ Data ();

    // Output preprocessed data to file
    void output (ofstream& out);
    void denseToArray(float *out);
    void sparseToArray3(int *out);
    void sparseToArray(int *out[], int batch);
    void sparseToBagArray(int *out[], int batch);
    void serialize (int *out);
    void deserialize (int *in);
};

inline bool file_exists (const char *fname);

void data_preprocess ();
void data_load (int numTrain, int numTest, vector<Data>& train_data, vector<Data>& test_data, int *num_features);

#endif // _DATA_H_
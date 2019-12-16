#ifndef _CUDNN_NMS_HPP_
#define _CUDNN_NMS_HPP_

#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>

using namespace std;

void filter_result_ver1(float *scores, float *boxes, vector<pair<int, int>> &filteredResult, const int numClasses, const int numBoxes, const int rowSize);
#endif

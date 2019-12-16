
#include "nms.hpp"
using namespace std;

/*
 * left_top (N * 2)
 * reight_bottom (N * 2)
 */
static inline void box_area(float *hw, float *left_top, float *right_bottom, int N)
{
	for(int i = 0; i < N; ++i) {
		for(int j = 0; j < 2; ++j) {
			right_bottom[(i * 2) + j] = right_bottom[(i * 2) + j] - left_top[(i * 2) + j];
			if(right_bottom[(i * 2 ) + j] < 0) right_bottom[(i * 2 ) + j] = 0;
		}
		hw[i] = right_bottom[(i * 2) + 0 ] * right_bottom[(i * 2) + 1];
	}
	return;
}

/*
 *
 * boxes0 (N * 4)
 * boxes1 (N * 4 or 1 * 4)
 *
 */
static inline void box_iou_ver1(int N, vector<pair<float, int>> &probs, float *boxes, int currentIdx, float *overlap_area)
{
	double eps = 1e-5;

	float *overlap_left_top = new float [N * 2];
	float *overlap_right_bottom = new float [N * 2];
	float *boxes0_lt = new float [N * 2];
	float *boxes0_rb = new float [N * 2];
	float *boxes1_lt = new float [1 * 2];
	float *boxes1_rb = new float [1 * 2];
	float *area0 = new float [N];
	float *area1 = new float [1];

	for(int i = 0; i < N; ++i) {
		for(int j = 0; j < 2; ++j) {
			int lt_idx = (probs[i].second * 4) + j;
			int rb_idx = lt_idx + 2;
			int cur_idx = (currentIdx * 4) + j;
			int half_idx = (i * 2) + j;
			int half_rb_idx = (i * 2) + j + 2;
			overlap_left_top[half_idx] = (boxes[lt_idx] > boxes[cur_idx]) ? (boxes[lt_idx]) : (boxes[cur_idx]);
			overlap_right_bottom[half_idx] = (boxes[rb_idx] < boxes[cur_idx + 2]) ? (boxes[rb_idx]) : (boxes[cur_idx + 2]);
			boxes0_lt[half_idx] = boxes[lt_idx];
			boxes0_rb[half_idx] = boxes[rb_idx];
			boxes1_lt[j] = boxes[cur_idx];
			boxes1_rb[j] = boxes[cur_idx + 2];
		}
	}

	box_area(overlap_area, overlap_left_top, overlap_right_bottom, N);
	box_area(area0, boxes0_lt, boxes0_rb, N);
	box_area(area1, boxes1_lt, boxes1_rb, 1);

	for(int i = 0; i < N; ++i) {
		area0[i] = area0[i] + area1[0] - overlap_area[i] + eps;
		overlap_area[i] /= area0[i];
	}

	delete overlap_left_top;
    delete overlap_right_bottom;
    delete boxes0_lt;
    delete boxes0_rb;
    delete boxes1_lt;
    delete boxes1_rb;
    delete area0;
	delete area1;
	return;
}

bool cmp(const pair<float, int> &a, const pair<float, int> &b) {
    return a.first > b.first;
}
void nms_ver1(float *scores, float *boxes, vector<pair<float, int>> &probs, vector<int> &picked)
{
	vector<pair<float, int>>::iterator it;
	float *iou = new float[probs.size()];
	float iou_threshold = 0.6;

	sort(probs.begin(), probs.end(), cmp);
	while(probs.size() > 0) {
		pair<float, int> current = probs[0];
		picked.push_back(current.second);
		if(probs.size() == 1)
			break;
		probs.erase(probs.begin());
        box_iou_ver1(probs.size(), probs, boxes, current.second, iou);
		int i = 0;
		for(it = probs.begin(); it != probs.end(); ++i) {
			if(iou[i] > iou_threshold) {
				it = probs.erase(it);
			} else {
				it++;
			}
		}
	}
	delete iou;
}

void filter_result_ver1(float *scores, float *boxes, vector<pair<int, int>> &filteredResult, const int numClasses, const int numBoxes, const int rowSize)
{
	vector<pair<float, int>> probs;
	vector<int> picked;
	const float score_threshold = 0.3;
	for (int class_index = 1; class_index < numClasses; ++class_index) {
		probs.clear();
		picked.clear();
		for(int i = 0; i < rowSize; ++i) {
			float val = scores[ (i * numClasses) + class_index ];
			if(val > score_threshold) {
				probs.push_back({val, i});
			}
		}
		if(probs.size() != 0 ) {
			nms_ver1(scores, boxes, probs, picked);
			for(auto p : picked) {
				filteredResult.push_back({class_index, p});
			}
		}
	}
}

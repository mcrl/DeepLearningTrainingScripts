#ifndef _RANDOM_H_
#define _RANDOM_H_

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC void get_normal_distribution(double mean, double stddev,
  int len, float *arr);
EXTERNC void get_xu_distribution(float *m, size_t nelem, float min, float max);

#endif /* _RANDOM_H_ */


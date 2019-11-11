#ifndef _DATASET_H_
#define _DATASET_H_

#include <stdio.h>

typedef struct dataset_s {
  FILE *fp_wav;
  FILE *fp_txt;

  int size;

  int *widths_wav;
  int *widths_txt;

  long *offsets_wav;
  long *offsets_txt;
} dataset_t;

dataset_t *dataset_create(const char *file_wav, const char *file_txt);
int dataset_get_wav(dataset_t *dataset, int index, float *out);
int dataset_get_txt(dataset_t *dataset, int index, int *out);
void dataset_destroy(dataset_t *dataset);

#endif


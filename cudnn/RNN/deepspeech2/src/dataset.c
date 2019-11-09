#include <stdlib.h>
#include <assert.h>

#include "dataset.h"
#include "params.h"

int read_int4(FILE *f)
{
  int val;
  assert(fread(&val, sizeof(int), 1, f) == 1);
  return val;
}

float read_float4(FILE *f)
{
  float val;
  assert(fread(&val, sizeof(float), 1, f) == 1);
  return val;
}

dataset_t *dataset_create(const char *file_wav, const char *file_txt)
{
  dataset_t *ret = (dataset_t *)malloc(sizeof(dataset_t));
  int err = 0;

  ret->fp_wav = fopen(file_wav, "rb");
  ret->fp_txt = fopen(file_txt, "rb");

  if (ret->fp_wav == NULL || ret->fp_txt == NULL) {
    free(ret);

    perror("fopen");
    return NULL;
  } 

  ret->size = read_int4(ret->fp_wav);
  assert(ret->size == read_int4(ret->fp_txt));

  fprintf(stderr, "dataset size : %d\n", ret->size);

  ret->widths_wav = (int *)malloc(sizeof(int) * ret->size);
  ret->widths_txt = (int *)malloc(sizeof(int) * ret->size);

  ret->offsets_wav = (long *)malloc(sizeof(long) * ret->size);
  ret->offsets_txt = (long *)malloc(sizeof(long) * ret->size);

  for (int i = 0; i < ret->size; i++) {
    ret->widths_wav[i] = read_int4(ret->fp_wav);
    ret->widths_txt[i] = read_int4(ret->fp_txt);

    ret->offsets_wav[i] = ftell(ret->fp_wav);
    ret->offsets_txt[i] = ftell(ret->fp_txt);

    err |= fseek(ret->fp_wav,
      sizeof(float) * ret->widths_wav[i] * FIXED_HEIGHT, SEEK_CUR);
    err |= fseek(ret->fp_txt, sizeof(int) * ret->widths_txt[i], SEEK_CUR); 

    assert(err == 0);
    if (i % 1000 == 0) {
      fprintf(stderr, "(%d / %d) complete\n", i, ret->size);
    }
  }

  fprintf(stderr, "read bytes: %ldB, %ldB\n", ftell(ret->fp_wav), ftell(ret->fp_txt));

  return ret;
}

int dataset_get_wav(dataset_t *dataset, int index, float *out)
{
  int err = fseek(dataset->fp_wav, dataset->offsets_wav[index], SEEK_SET);
  assert(err == 0);

  err = fread((void *)out, sizeof(float),
    dataset->widths_wav[index] * FIXED_HEIGHT, dataset->fp_wav);
  assert(err == dataset->widths_wav[index] * FIXED_HEIGHT);

  return 0;
}

int dataset_get_txt(dataset_t *dataset, int index, int *out)
{
  int err = fseek(dataset->fp_txt, dataset->offsets_txt[index], SEEK_SET);
  assert(err == 0);

  err = fread((void *)out, sizeof(int), dataset->widths_txt[index], dataset->fp_txt);
  assert(err == dataset->widths_txt[index]);

  return 0;
}

void dataset_destroy(dataset_t *dataset)
{
  fclose(dataset->fp_wav);
  fclose(dataset->fp_txt);

  free(dataset->widths_wav);
  free(dataset->widths_txt);

  free(dataset->offsets_wav);
  free(dataset->offsets_txt);

  free(dataset);
}


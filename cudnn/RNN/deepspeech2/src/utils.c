#include "utils.h"

void **get_global_workspace(size_t bytes)
{
  static size_t curr_bytes = 0;
  static void *workspace = NULL;

  if (curr_bytes < bytes || workspace == NULL) {
    if (workspace != NULL) chkCUDA(cudaFree(workspace));
    curr_bytes = bytes;
    chkCUDA(cudaMalloc((void **)&workspace, bytes));
  }

  return &workspace;
}


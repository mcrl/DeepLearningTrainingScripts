#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <mpi.h>

#include "params.h"
#include "deepspeech.h"
#include "dataset.h"
#include "utils.h"

dataset_t *dataset;

dataset_t *load_input(const char *file_wav, const char *file_txt)
{
  return dataset_create(file_wav, file_txt);
}

int main(int argc, char *argv[])
{
  int world_rank, world_size;
  int num_dev_per_node = 0;
  int is_train = 1;

#ifdef DEBUG
  fprintf(stderr, "It's debug mode!\n");
#endif

  if (argc != 2 && argc != 3 && argc != 5) {
    fprintf(stderr, "./deepspeech.cudnn num_dev_per_node"
      " [train | infer, default = train]"
      " [path_wav_bin path_txt_bin, default ="
      " data/{train | val}_wav.bin"
      " data/{train | val}_txt.bin]\n");
    return 1;
  }

  MPI_CALL(MPI_Init(NULL, NULL));

  MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

  num_dev_per_node = atoi(argv[1]);
  assert(num_dev_per_node > 0);

  if (world_rank == 0) {
    fprintf(stderr, "world_size : %d, num_dev_per_node : %d\n",
      world_size, num_dev_per_node);
  }

  if (argc >= 3) {
    is_train = (strncmp(argv[2], "train", 5) == 0);
  }

  if (argc == 5) {
    dataset = load_input(argv[3], argv[4]);
  }
  else if (is_train == 1) {
    dataset = load_input("data/train_wav.bin",
      "data/train_txt.bin");
  }
  else {
    dataset = load_input("data/val_wav.bin",
      "data/val_txt.bin");
  }

  if (dataset == NULL) {
    fprintf(stderr, "failed to load dataset\n");
  }

  if (is_train == 1) {
    deepspeech_train(dataset, world_rank, world_size, num_dev_per_node);
  }
  else {
    deepspeech_eval(dataset, world_rank, world_size, num_dev_per_node);
  }

  MPI_CALL(MPI_Finalize());

  return 0;
}


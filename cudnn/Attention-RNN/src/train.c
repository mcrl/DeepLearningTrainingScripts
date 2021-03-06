#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

#include "options.h"
#include "dataset.h"
#include "snudnn.h"
#include "optimizer.h"
#include "tensor.h"
#include "gpu.h"


#include "loss.h"
#include "utils.h"
#include "cache_allocator.h"

static struct options options;

int main(int argc, char *argv[])
{
	snudnn_init(argc, argv);
	if (rank == 0)
		printf("Using %d GPUs\n", nrank);

	dataset_t *dataset = dataset_training(
			"data/wmt17_en_de/train.en-de.en",
			"data/wmt17_en_de/train.en-de.de",
			"data/wmt17_en_de/dict.en.txt",
			"data/wmt17_en_de/dict.de.txt");

	parse_opts(&options, argc, argv, dataset);

	size_t free, total;
	chkCUDA(cudaMemGetInfo(&free, &total));
	cacher_init((size_t)((double)free * 0.6));


	model_t *model = model_create(&options);
	loss_t *loss = loss_create(model);
	int max_epoch = options.train.max_epoch;
	int nbatch = dataset_nbatch(dataset);
	double total_time = 0.0;

	struct batch_t batch;
	int N;
	loss_info_t loss_info;
	for (int epoch = 0; epoch < max_epoch; epoch++) {
		for (int i = 0; i < nbatch; i++) {
			dataset_load_batch(dataset, &batch, i);

			tensor_t *target = batch.target;
			N = batch.size * (batch.max_src_len > batch.max_tgt_len ? batch.max_src_len : batch.max_tgt_len);

			struct timespec st, ed;
			chkCUDA(cudaDeviceSynchronize());
			clock_gettime(CLOCK_MONOTONIC, &st); // start timer

			tensor_t *out = model_forward(model, &batch);
			loss_info = loss_forward(loss, out, target);
			model_zerograd(model);
			loss_backward(loss, out, target);
			model_update(model, N);

			chkCUDA(cudaDeviceSynchronize());
			clock_gettime(CLOCK_MONOTONIC, &ed); // end timer

			double iter_time = (ed.tv_sec - st.tv_sec) * 1000 + (ed.tv_nsec - st.tv_nsec) / 1000000.0;
			total_time += iter_time;
			if (rank == 0) {
				fprintf(stderr, "[%d/%d] batch_size: %d, src_len: %d, tgt_len: %d, loss: %lf,",
						i+1, nbatch, batch.size, batch.max_src_len, batch.max_tgt_len, loss_info.loss);
				fprintf(stderr, " iter: %fms, ms/iter: %f, total: %f ms\n", iter_time, total_time / (i+1) / nrank, total_time);
			}
			dataset_drop_batch(dataset, i);
		}
	}

	return 0;
}

#include <assert.h>

#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>

#include "dataset.h"
#include "optimizer.h"
#include "gpu.h"
#include "model.h"
#include "options.h"
#include "utils.h"

#include "cache_allocator.h"


struct _model_t {
	encoder_t *encoder;
	decoder_t *decoder;

	float *parameter;
	float *dparameter;
	size_t nparams;
};

static size_t model_params(model_t *model)
{
	size_t nparams = encoder_params(model->encoder);
	nparams += decoder_params(model->decoder);
	return nparams;
}

static void model_init_params(model_t *model)
{
	size_t nparams = model_params(model);
	float *buf;

	// allocate both parameter and gradient buffers
	chkCUDA(cudaMalloc((void**)&buf, 2 * nparams * sizeof(float)));

	model->parameter = buf;
	model->dparameter = buf + nparams;

	size_t offset = encoder_init_params(model->encoder,
			model->parameter,
			model->dparameter
			);
	offset += decoder_init_params(model->decoder,
			model->parameter + offset,
			model->dparameter + offset
			);

	assert(offset == nparams);

	model->nparams = nparams;
}

static void model_init(model_t *model, struct options *options)
{
	model->encoder = encoder_create(&options->encoder);
	model->decoder = decoder_create(options->decoder);

	size_t nparams = model_params(model);
	if (rank == 0)
		printf("%lu parameters\n", nparams);
	model_init_params(model);
}

static void model_release(model_t *model)
{
	decoder_free(model->decoder);
	encoder_free(model->encoder);
}

model_t* model_create(struct options *options)
{
	model_t *model = malloc(sizeof(model_t));
	model_init(model, options);
	return model;
}

void model_free(model_t *model)
{
	model_release(model);
	free(model);
}

tensor_t* model_forward(model_t *model, struct batch_t *batch)
{
	tensor_t *input = batch->input;
	tensor_t *target = batch->target;

	tensor_t *decoder_out;
	tensor_quadruple_t encoder_out;

	encoder_out = encoder_forward(model->encoder, input, batch->src_len_array);
	decoder_out = decoder_forward(model->decoder, target, encoder_out.first, encoder_out.second, encoder_out.third, encoder_out.fourth);

	return decoder_out;
}

void model_clear(model_t *model)
{
	encoder_clear(model->encoder);
	decoder_clear(model->decoder);
	cacher_clear();
}

tensor_t* model_inference(model_t *model, struct batch_t *batch)
{
	tensor_t *input = batch->input;
	tensor_t *target = batch->target;

	tensor_t *decoder_out;
	tensor_quadruple_t encoder_out;

	encoder_out = encoder_inference(model->encoder, input, batch->src_len_array);
	decoder_out = decoder_inference(model->decoder, target, encoder_out.first, encoder_out.second, encoder_out.third, encoder_out.fourth);

	return decoder_out;
}

void model_zerograd(model_t *model)
{
	cudaMemsetAsync(model->dparameter, 0, model->nparams * sizeof(float), update_stream);
}

void model_backward(model_t *model, tensor_t *dy)
{
	tensor_triple_t dy_encoder;

	dy_encoder = decoder_backward(model->decoder, dy);

	encoder_backward(model->encoder, dy_encoder.first, dy_encoder.second, dy_encoder.third);

	model_clear(model);
}

void model_update(model_t *model, int N)
{
	optimizer_step(model->dparameter, model->parameter, model->nparams, N);
}

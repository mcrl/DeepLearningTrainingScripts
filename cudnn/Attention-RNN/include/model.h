#ifndef __MODEL_H__
#define __MODEL_H__

#include "tensor.h"

#include "dataset.h"
typedef struct _model_t model_t;
struct batch_t;
struct options;

//
// create model object
// 
model_t* model_create(struct options *options);

//
// destroy model object
// 
void model_free();

//
// forward for training
// 
tensor_t* model_forward(model_t *model, struct batch_t *batch);

//
// forward for inference
// 
tensor_t* model_inference(model_t *model, struct batch_t *batch);

//
// set gradients to zero
// 
void model_zerograd(model_t *model);

//
// backward, must be called after 'model_zerograd' is called
// 
void model_backward(model_t *model, tensor_t *dy);

//
// update graidents to weights.
// must be called after model_backward is called
// 
void model_update(model_t *model, int N);

//
// release temporary resources.
// this function is used to reclaim for each iteration
// 
void model_clear(model_t *model);

#endif //__MODEL_H__

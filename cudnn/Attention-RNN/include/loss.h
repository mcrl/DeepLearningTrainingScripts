#ifndef __LOSS_H_
#define __LOSS_H_

#include "model.h"

typedef struct _loss_t loss_t;
typedef struct {
	double loss;
	double nll_loss;
} loss_info_t;

loss_t* loss_create(model_t *model);
void    loss_free(model_t *model);
loss_info_t    loss_forward(loss_t *loss, tensor_t *out, tensor_t *target);
loss_info_t    loss_inference(loss_t *loss, tensor_t *out, tensor_t *target);
tensor_t* loss_backward(loss_t *loss, tensor_t *out, tensor_t *target);

#endif //__LOSS_H_

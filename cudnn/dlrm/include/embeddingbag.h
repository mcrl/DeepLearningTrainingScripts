#ifndef _EMBEDDINGBAG_H_
#define _EMBEDDINGBAG_H_

#include "configs.h"
#include "tensor.h"
#include "utils.h"

class EmbeddingBag {
public:
    int batch_size, rows, vector_size, bag_size;
    float *table;
    int ndev;

    int *in; float *delta;
    int *gatheredIn; float *gatheredDelta;

    EmbeddingBag (int batch_size_, int rows_, int bag_size_, int vector_size_, int ndev_, bool init);

    void forward (IntegerTensor *t_in, Tensor *t_out);
    void backward (IntegerTensor *t_in, Tensor *t_out, Tensor *t_out_grad);
    void update ();
};

#endif // _EMBEDDINGBAG_H_
#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include "configs.h"
#include "tensor.h"

class ActivationLayer {
public:
    int ndev;
    cudnnActivationDescriptor_t actDesc;

    ActivationLayer (int ndev_, const char* type);
    void forward (Tensor *t_input, Tensor *t_output);
    void backward (Tensor *t_input, Tensor *t_inputGrad, Tensor *t_output, Tensor *t_outputGrad);
};

#endif // _ACTIVATION_H_

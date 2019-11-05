#ifndef _FC_H_
#define _FC_H_

#include "configs.h"
#include "tensor.h"
#include "utils.h"

class FCLayer {
public:

    cudnnConvolutionDescriptor_t convDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionFwdAlgo_t convFwdAlgo;
    cudnnConvolutionBwdFilterAlgo_t convBwdFilterAlgo;
    cudnnConvolutionBwdDataAlgo_t convBwdDataAlgo;

    float *d_weight, *d_weightDelta;
    Tensor *bias, *biasDelta;

    int workspaceBytes = 0;
    float *d_workspace;

    int inputSize, outputSize, batch_size;
    int ndev;

    FCLayer (int inputs, int outputs, int batches, int ndev_, bool init);
    
    ~FCLayer ();

    // t_output is the output of this function
    void forward (Tensor *t_input, Tensor *t_output);
    // t_inputGrad is the output of this function
    void backward (Tensor *t_input, Tensor *t_inputGrad, Tensor *t_outputGrad);

    /* update weight & bias based on gradient calc with backward() call */
    void update ();
};

#endif // _FC_H_

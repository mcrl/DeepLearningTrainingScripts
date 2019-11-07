
#include "activation.h"

ActivationLayer::ActivationLayer (int ndev_, const char* type) {
    ndev = ndev_;
    CUDNN_CALL( cudnnCreateActivationDescriptor(&actDesc) );

    if ( strcmp(type, "sigmoid") == 0 ) {
        CUDNN_CALL ( cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0.0));
    }
    else if ( strcmp(type, "relu") == 0 ) {
        CUDNN_CALL ( cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));
    }
}

void ActivationLayer::forward (Tensor *t_input, Tensor *t_output) {
    CUDA_CALL( cudaSetDevice(ndev) );
    CUBLAS_CALL( cublasSetPointerMode(cublas[ndev], CUBLAS_POINTER_MODE_HOST) );
    CUDNN_CALL( cudnnActivationForward(
        cudnn[ndev], actDesc,
        &one, t_input->desc, t_input->d_mem,
        &zero, t_output->desc, t_output->d_mem
    ));
}

void ActivationLayer::backward (Tensor *t_input, Tensor *t_inputGrad, Tensor *t_output, Tensor *t_outputGrad) {
    CUDA_CALL( cudaSetDevice(ndev) );
    CUDNN_CALL( cudnnActivationBackward(
        cudnn[ndev], actDesc,
        &one, t_output->desc, t_output->d_mem,
        t_outputGrad->desc, t_outputGrad->d_mem,
        t_input->desc, t_input->d_mem,
        &zero, t_inputGrad->desc, t_inputGrad->d_mem
    ));
}

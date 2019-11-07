#include "fc.h"


FCLayer::FCLayer (int inputs, int outputs, int batches, int ndev_, bool init) :
            inputSize(inputs), outputSize(outputs), batch_size(batches), ndev(ndev_) {

    CUDNN_CALL( cudnnCreateFilterDescriptor(&filterDesc) );
    CUDNN_CALL( cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        outputSize, inputSize, 1, 1));

    CUDNN_CALL( cudnnCreateConvolutionDescriptor(&convDesc) );
    CUDNN_CALL( cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    convBwdDataAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    convBwdFilterAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    convFwdAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    workspaceBytes = 1024 * 1024 * sizeof(float);
    CUDA_CALL( cudaMalloc(&d_weight, inputSize * outputSize * sizeof(float)) );
    CUDA_CALL( cudaMalloc(&d_weightDelta, inputSize * outputSize * sizeof(float)) );
    CUDA_CALL( cudaMalloc(&d_workspace, workspaceBytes) );

    bias = new Tensor(1, outputSize, 1, 1, ndev);
    biasDelta = new Tensor(1, outputSize, 1, 1, ndev);

    if ( init ) {
        initRand(bias->d_mem, outputSize, sqrt(1.0 / outputSize), ndev);
        initRand(d_weight, inputSize * outputSize, sqrt(2.0 / (inputSize + outputSize)), ndev);
    }
}

FCLayer::~FCLayer () {

}


void FCLayer::forward (Tensor *t_input, Tensor *t_output) {

    CUDA_CALL( cudaSetDevice(ndev) );

    // apply convolution weight
    CUDNN_CALL( cudnnConvolutionForward(
        cudnn[ndev],
        &one, t_input->desc, t_input->d_mem,  // input
        filterDesc, d_weight,                 // weight
        convDesc, convFwdAlgo, // conv
        d_workspace, workspaceBytes,
        &zero, t_output->desc, t_output->d_mem
    ));

    // apply bias
    CUDNN_CALL( cudnnAddTensor(
        cudnn[ndev],
        &one, bias->desc, bias->d_mem,
        &one, t_output->desc, t_output->d_mem
    ));

}

// t_inputGrad is the output of this function
void FCLayer::backward (Tensor *t_input, Tensor *t_inputGrad, Tensor *t_outputGrad) {

    CUDA_CALL( cudaSetDevice(ndev) );

    // Data grad
    CUDNN_CALL( cudnnConvolutionBackwardData(
        cudnn[ndev],
        &one, filterDesc, d_weight,
        t_outputGrad->desc, t_outputGrad->d_mem,
        convDesc, convBwdDataAlgo,
        d_workspace, workspaceBytes,
        &zero, t_inputGrad->desc, t_inputGrad->d_mem
    ));

    // Filter delta
    CUDNN_CALL( cudnnConvolutionBackwardFilter(
        cudnn[ndev],
        &one, t_input->desc, t_input->d_mem,
        t_outputGrad->desc, t_outputGrad->d_mem,
        convDesc, convBwdFilterAlgo,
        d_workspace, workspaceBytes,
        &zero, filterDesc, d_weightDelta
    ));

    // Bias delta
    CUDNN_CALL( cudnnConvolutionBackwardBias(
        cudnn[ndev],
        &one, t_outputGrad->desc, t_outputGrad->d_mem,
        &zero, biasDelta->desc, biasDelta->d_mem
    ));
}

/* update weight & bias based on gradient calc with backward() call */
void FCLayer::update () {

    CUDA_CALL( cudaSetDevice(ndev) );

    CUBLAS_CALL( cublasSetPointerMode(cublas[ndev], CUBLAS_POINTER_MODE_HOST) );

    // update filter
    CUBLAS_CALL( cublasSaxpy(
        cublas[ndev], inputSize * outputSize,
        &lr, d_weightDelta, 1,
        d_weight, 1
    ));

    // update bias
    CUBLAS_CALL( cublasSaxpy(
        cublas[ndev], outputSize,
        &lr, biasDelta->d_mem, 1,
        bias->d_mem, 1
    ));
}


# Referecne-benchmark (Inference only)
There is a benchamrk for PyTorch.

* ssd-mobilenet v1


## Requirements

* Anaconda
* PyTorch
* TensorFlow
* CUDA
* cuDNN

## Setup the environment

1. Install anaconda
2. Crate env, activate and install common packages.
```
conda create -n benchmark_3.6 python=3.6
conda activate benchmark_3.6
pip install tensorflow-gpu
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```
3. Install the additional packages required to specific benchmark scripts.
```
pip install scipy imageio cython
conda install opencv
#install cocoapi for ssd-mobilenet
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
```

## How to run the benchmark
1. Enter the benchmark directory you want to run.
2. Please download the coco 2017 validaion dataset by refering "geting input dataset".
3. Modify the dataset path in run.sh.
4. Run run.sh

## Source from

### PyTorch
* ssd-mobilenet
  * reference : (apache-2.0) https://github.com/mlperf/inference/tree/master/others/edge/object_detection/ssd_mobilenet/pytorch
  * pre-trained model : (apache-2.0) http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
  * getting input dataset : refer to "Install the COCO 2017 validation dataset (5,000 images)" section.
                 (https://github.com/mlperf/inference/blob/master/v0.5/classification_and_detection/optional_harness_ck/detection/README.md)
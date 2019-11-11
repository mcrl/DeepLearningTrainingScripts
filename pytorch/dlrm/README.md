Deep Learning Recommendation Model for Personalization and Recommendation Systems:
=================================================================================
*Copyright (c) Facebook, Inc. and its affiliates.*

This readme is from the [original DLRM repository](https://github.com/facebookresearch/dlrm), with some modifications.

How to run
--------------------

The code supports interface with the [Criteo Kaggle Display Advertising Challenge Dataset](https://labs.criteo.com/2014/09/kaggle-contest-dataset-now-available-academic-use/).

Please do the following to prepare the dataset for use with DLRM code:

- First, specify the raw data file (train.txt) as downloaded with `--raw-data-file=<data/train.txt>`
- This is then pre-processed (categorize, concat across days...) to allow using with dlrm code
- The processed data is stored as `*.npz` file in `<root_dir>/data/*.npz`
- The processed file `*.npz` can be used for subsequent runs with `--processed-data-file=<data/*.npz>`

Then, run with the following script.

 ```
 ./dlrm_s_criteo_kaggle.sh
 ```

This will run inference & training on multiple GPUs (if available). Please check lines 254-258 and 712-718 in `dlrm_s_pytorch.py`. This code uses custom parallel-forward method for multi-GPU.


Model checkpoint saving/loading
-------------------------------
During training, the model can be saved using --save-model=<path/model.pt>

The model is saved if there is an improvement in test accuracy (which is checked at --test-freq intervals).

A previously saved model can be loaded using --load-model=<path/model.pt>

Once loaded the model can be used to continue training, with the saved model being a checkpoint.
Alternatively, the saved model can be used to evaluate only on the test data-set by specifying --inference-only option.



Implementation
--------------
**DLRM PyTorch**. Implementation of DLRM in PyTorch framework:

       dlrm_s_pytorch.py

**DLRM Data**. Implementation of DLRM data generation and loading:

       dlrm_data_pytorch.py, data_utils.py

Version
-------
0.1 : Initial release of the DLRM code

Requirements
------------
pytorch-nightly (*6/10/19*)

onnx (*optional*)

torchviz (*optional*)

License
-------
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

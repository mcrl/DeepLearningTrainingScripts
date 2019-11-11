Deep Learning Recommendation Model for Personalization and Recommendation Systems:
=================================================================================
*Copyright (c) Facebook, Inc. and its affiliates.*

This readme is from the [original DLRM repository](https://github.com/facebookresearch/dlrm), with some modifications.

How to run
--------------------

0)
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


1) A sample run of the code, with a tiny model is shown below
```
$ python dlrm_s_pytorch.py --mini-batch-size=2 --data-size=6
time/loss/accuracy (if enabled):
Finished training it 1/3 of epoch 0, -1.00 ms/it, loss 0.451893, accuracy 0.000%
Finished training it 2/3 of epoch 0, -1.00 ms/it, loss 0.402002, accuracy 0.000%
Finished training it 3/3 of epoch 0, -1.00 ms/it, loss 0.275460, accuracy 0.000%
```
2) A sample run of the code, with a tiny model in debug mode
```
$ python dlrm_s_pytorch.py --mini-batch-size=2 --data-size=6 --debug-mode
model arch:
mlp top arch 3 layers, with input to output dimensions:
[8 4 2 1]
# of interactions
8
...
(more lines below)
```


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

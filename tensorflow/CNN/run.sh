#!/bin/bash

if [ $# -ne 1 ]; then
  echo " Usage: $0 [model=vgg_16,inception_v4,resnet_v2_152,densenet_169]"
  exit 1
fi

trainer=train_image_classifier.py
model=$1

# temporary directories
train_root=$HOME/slim_tmp/train
log_root=$HOME/slim_tmp/log

# imagenet data (in TFRecord format) directory
imagenet_dir=/media/e2/TensorFlow_Data/ImageNetData

mkdir -p ${train_root}/${model} && mkdir -p ${log_root}/${model}
./${trainer} --train_dir=${train_root}/${model} --dataset_name=imagenet --dataset_split_name=train --dataset_dir=${imagenet_dir} --model_name=${model} --labels_offset=1

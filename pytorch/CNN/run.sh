#!/bin/bash

####################################################

batch_size=16
iteration=100

host_list=("c0" "c1" "c2" "c3" "c4" "c5" "c6" "c8")

dataset_dir=/home/tuan/imagenet_data/CLS-LOC

####################################################

gen_device_config() {
  if [ $1 -eq 1 ] ; then
    echo 'CUDA_VISIBLE_DEVICES=0 '
  elif [ $1 -eq 2 ] ; then
    echo 'CUDA_VISIBLE_DEVICES=0,1 '
  elif [ $1 -eq 3 ] ; then
    echo 'CUDA_VISIBLE_DEVICES=0,1,2 '
  elif [ $1 -eq 4 ] ; then
    echo 'CUDA_VISIBLE_DEVICES=0,1,2,3 '
  else
    echo 'CUDA_VISIBLE_DEVICES= '
  fi
}

run_test() {
  local num_nodes=$1
  local num_gpus=$2
  local net=$3

  local total_batch_size=$((num_nodes * num_gpus * batch_size))
  local device_config=$(gen_device_config ${num_gpus})
  local log_file=log/${net}_${num_nodes}_${num_gpus}

  echo "[${net}] ${num_nodes} nodes x ${num_gpus} gpus"

  if [ ${num_nodes} -eq 1 ] ; then
    if [ ${num_gpus} -eq 1 ] ; then
      export ${device_config}
      python3.6 ${trainer} \
        --workers 0 \
        --batch-size ${total_batch_size} \
        --iter ${iteration} \
        --epochs 1 \
        --arch ${net} \
        --lr 0.01 \
        --momentum 0 \
        --weight-decay 0 \
        --world-size 1 \
        --rank 0 \
        --pretrained \
        --evaluate \
        ${dataset_dir} \
        > ${log_file}
    else
      export ${device_config}
      python3.6 ${trainer} \
        --workers 0 \
        --batch-size ${total_batch_size} \
        --iter ${iteration} \
        --epochs 1 \
        --arch ${net} \
        --lr 0.01 \
        --momentum 0 \
        --weight-decay 0 \
        --dist-url 'tcp://127.0.0.1:12345' \
        --dist-backend 'nccl' \
        --multiprocessing-distributed \
        --world-size 1 \
        --rank 0 \
        --pretrained \
        --evaluate \
        ${dataset_dir} \
        > ${log_file}
    fi
  else
    export NCCL_TREE_THRESHOLD=0
    python3.6 ${trainer} \
      --workers 0 \
      --batch-size ${total_batch_size} \
      --iter ${iteration} \
      --epochs 1 \
      --arch ${net} \
      --lr 0.01 \
      --momentum 0 \
      --weight-decay 0 \
      --dist-url 'tcp://192.168.0.10${master_id}:12345' \
      --dist-backend 'nccl' \
      --multiprocessing-distributed \
      --world-size ${num_nodes} \
      --rank ${local_id} \
      --pretrained \
      --evaluate \
      ${dataset_dir} \
      > ${log_file}
  fi
}

####################################################

if [ $# -eq 0 ] ; then
  master_id=0
  local_id=0
  num_nodes=1
elif [ $# -ne 3 ] ; then
  echo " Usage: $0 [master_id=0] [local_id=0] [num_nodes=1]"
  exit 1
else
  master_id=$1
  local_id=$2
  num_nodes=$3
fi


trainer=main_modified.py

mkdir -p log

for net in vgg16 resnet50 densenet121 inception_v3 ; do
  if [ ${num_nodes} -eq 1 ] ; then
    run_test 1 1 ${net}
  else
    run_test ${num_nodes} 4 ${net}
  fi
done

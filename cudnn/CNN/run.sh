#!/bin/bash

####################################################

batch_size=16
iteration=21

host_list=("c0" "c1" "c2" "c3" "c4" "c5" "c6" "c8")
dev_list=("3" "2" "1" "0")

####################################################

host_name() {
  local num_nodes=$1
  local acc=${host_list[0]}
  for (( i = 1 ; i < ${num_nodes} ; i++ )) ; do
    acc="${acc},${host_list[${i}]}"
  done
  echo ${acc}
}

dev_name() {
  local num_devs=$1
  local acc=${dev_list[0]}
  for (( i = 1 ; i < ${num_devs} ; i++ )) ; do
    acc="${acc},${dev_list[${i}]}"
  done
  echo ${acc}
}

run_test() {
  local num_nodes=$1
  local num_gpus=$2
  local net=$3

  local total_batch_size=$((num_nodes * num_gpus * batch_size))
  local hosts=$(host_name ${num_nodes})
  local devs=$(dev_name ${num_gpus})

  echo "[${net}] ${num_nodes} nodes x ${num_gpus} gpus"
  
  mpirun \
    --host ${hosts} -x CUDA_VISIBLE_DEVICES=${devs} \
    ./${net} ${total_batch_size} ${iteration} ${data_path} ${label_path} \
    model/${net}_${total_batch_size}_${iteration} > \
    log/${net}_${num_nodes}_${num_gpus}
}

####################################################

mkdir -p model
mkdir -p obj
mkdir -p log

for net in vgg resnet densenet inception ; do
data_path=/home/data/imagenet_10240_${net}.data
label_path=/home/data/imagenet_10240_${net}.label
make ${net}
run_test 1 1 ${net}
run_test 1 2 ${net}
run_test 1 4 ${net}
run_test 2 4 ${net}
run_test 4 4 ${net}
#run_test 8 4 ${net}
done

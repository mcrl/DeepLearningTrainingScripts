# Fairseq implementation of Attention-RNN (https://github.com/pytorch/fairseq)


0. How to compile
	cd fairseq
	pip install --editable .

1. Dataset
	# Download and prepare the data
	cd examples/translation/
	# WMT'17 data:
	bash prepare-wmt14en2de.sh
	# or to use WMT'14 data:
	# bash prepare-wmt14en2de.sh --icml17
	cd ../..
	
	# Binarize the dataset
	TEXT=examples/translation/wmt17_en_de
	fairseq-preprocess \
	--source-lang en --target-lang de \
	--trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
	--destdir data-bin/wmt17_en_de --thresholdtgt 0 --thresholdsrc 0 \
	--workers 20

2. How to run

	(note) change --master_addr, --master_port, --nnodes, --node_rank options appropriately

	a) single node
	CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/wmt17_en_de --momentum 0 --lr 1 --clip-norm 0.1 --max-tokens 1200  --criterion cross_entropy --lr-scheduler fixed --curriculum 1 --fix-batches-to-gpus --optimizer sgd --distributed-world-size 1 --arch lstm_luong_wmt_en_de --save-dir checkpoints/run4 --ddp-backend=no_c10d --validate-interval 100000 --no-save --max-epoch 1

	b) multi-node (master)
	NCCL_TREE_THRESHOLD=0 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=0 --master_addr="192.168.10.100" --master_port=22549 $(which fairseq-train) data-bin/wmt17_en_de --momentum 0 --lr 1 --clip-norm 0.1 --max-tokens 1200 --criterion cross_entropy --lr-scheduler fixed --curriculum 1 --fix-batches-to-gpus --optimizer sgd --arch lstm_luong_wmt_en_de --save-dir checkpoints/run4 --ddp-backend=no_c10d --validate-interval 100000 --no-save --max-epoch 1 --distributed-no-spawn

	c) multi-node (slave)
	NCCL_TREE_THRESHOLD=0 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=3 --master_addr="192.168.10.100" --master_port=22549 $(which fairseq-train) data-bin/wmt17_en_de --momentum 0 --lr 1 --clip-norm 0.1 --max-tokens 1200 --criterion cross_entropy --lr-scheduler fixed --curriculum 1 --fix-batches-to-gpus --optimizer sgd --arch lstm_luong_wmt_en_de --save-dir checkpoints/run4 --ddp-backend=no_c10d --validate-interval 100000 --no-save --max-epoch 1 --distributed-no-spawn

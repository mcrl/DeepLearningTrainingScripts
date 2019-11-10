# Fairseq implementation of Attention-RNN (https://github.com/pytorch/fairseq)

0. Dataset
	Preprocessed wmt17 dataset is already in './fairseq/data-bin/' directory.

1. How to compile
	cd fairseq
	pip install --editable .

2. How to run

a) training
	./train_attention_rnn.sh [number of processes]

b) inference
	./inference_attention_rnn.sh [number of processes]

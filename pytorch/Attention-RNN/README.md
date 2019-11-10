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

a) training
	./train_attention_rnn.sh [number of processes]

b) inference
	./inference_attention_rnn.sh [number of processes]

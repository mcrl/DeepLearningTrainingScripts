0. How to compile

	make [all]

1. Dataset

	Preprocessed wmt17 dataset is already in './data/' directory.

2. How to run

(note) 1 mpi process for each GPUs.

a) training
	mpirun -np [# processes] --host [host names] ./attention-rnn-training

b) inference
	mpirun -np [# processes] --host [host names] ./attention-rnn-inference

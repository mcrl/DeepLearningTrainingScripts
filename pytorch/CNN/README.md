# CNN training scripts in PyTorch

modified from PyTorch official example

## Original repo
https://github.com/pytorch/examples/tree/master/imagenet

## Execution examples

### Single node, single GPU:
```bash
python main_modified.py -a inception_v3 --lr 0.1 [imagenet-folder with train and val folders]
```

### Single node, multiple GPUs:
```bash
python main_modified.py -a inception_v3 --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 [imagenet-folder with train and val folders]
```

### Multiple nodes:

Node 0:
```bash
python main_modified.py -a inception_v3 --dist-url 'tcp://IP_OF_NODE0:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0 [imagenet-folder with train and val folders]
```

Node 1:
```bash
python main_modified.py -a inception_v3 --dist-url 'tcp://IP_OF_NODE0:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1 [imagenet-folder with train and val folders]
```

# CNN training scripts in cuDNN

## data
벤치마크 목적으로 개발하여 데이터 전처리 기능은 따로 없으므로, 임의의 데이터셋을 사용하려면 custom python script를 작성하여 직접 전처리를 해주어야 합니다.
Pytorch의 dataloader가 전처리를 완료한 이미지, 라벨 데이터를 그대로 덤프하여 이용합니다.
아래의 의사 코드를 참고하십시오.

```
data, label = train_loader(...) # configured as 3200 images

with open("imagenet_3200.data", "w") as f:
  data.numpy().tofile(f)

with open("imagenet_3200.label", "w") as f:
  label.numpy().tofile(f)
```

data/ 디렉터리의 imagenet dataset binary file을 참고하십시오.

## compile
Makefile이 마련되어 있습니다.
라이브러리 : cuDNN, cuBLAS, MPI의 설치 경로를 알맞게 지정하여 주십시오. 
아래의 수정 실시예를 참고하십시오.

```
# MPI 설치 경로 설정
CC_INCLUDE := -I$(INCLUDE_PATH) -I/usr/include/openmpi-x86_64
CC_LIBRARY := -lm -L/usr/lib64/openmpi/lib -lmpi

# cuda 런타임 및 cuBLAS, cuDNN 설치 경로 설정
CC_CUDA_INCLUDE := -I/usr/include -I/usr/local/cuda/include -I/usr/local/cuda/cudnn-7.4.2/include
CC_CUDA_LIBRARY := -L/usr/lib/x86_64-linux-gnu/ -L/usr/lib64 -L/usr/local/cuda/lib64/ -L/usr/local/cuda/cudnn-7.4.2/lib64 -lcudnn -lcublas -lcudart -lnccl
CUDA_INCLUDE_PATH := -I/usr/local/cuda/include -I/usr/local/cuda/cudnn-7.4.2/include
```

기본적으로 training 모드로 컴파일됩니다.
inference 모드로 컴파일하기 위해서는 다음 라인을 주석처리하십시오.

```
#DEFINES += -DUSE_TRAINING
```

### compile vgg
$ make vgg

### compile resnet
$ make resnet

### compile densenet
$ make densenet

### compile inception
$ make inception

### cleanup
$ make clean

## run
run.sh이 마련되어 있습니다.
실행 환경에 맞추어서 스크립트를 수정하여 사용하기를 권장합니다.

스크립트의 host\_list를 클러스터 구성에 맞게 지정해주십시오.
스크립트의 dev\_list를 노드 구성에 맞게 지정해주십시오.
스크립트의 메인 루프문을 수정하여 실험을 설계합니다.

1. 실행할 네트워크 이름을 알맞게 지정해주십시오.
1. 데이터셋 덤프 파일의 경로를 알맞게 지정해주십시오.
1. run\_test 함수에 알맞은 인자를 지정해주십시오.

[ run\_test {number of nodes} {number of devices} {model name} ]

예를 들어 GPU가 2개씩 설치된 노드 4대를 이용하여 `inception-v3` 네트워크 스크립트를 실행하는 다음 메인 루프문을 다음과 같이 작성할 수 있습니다.

```
for net in inception ; do
  data_path=data/imagenet_3200.data
  label_path=data/imagenet_3200.label
  make ${net}
  run_test 4 2 ${net}
done
```

실행 결과는 log/ 디렉터리에 텍스트 파일로 저장됩니다.

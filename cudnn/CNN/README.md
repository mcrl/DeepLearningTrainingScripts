# CNN training scripts in cuDNN

## data
벤치마크 목적으로 개발하여 데이터 전처리 기능은 따로 없으므로, 임의의 데이터셋을 사용하려면 custom python script를 작성하여 직접 전처리를 해주어야 합니다.
Pytorch의 dataloader가 전처리를 완료한 이미지, 라벨 데이터를 그대로 덤프하여 이용합니다.
data/ 디렉터리의 imagenet dataset binary file을 참고하십시오.

## compile
Makefile이 마련되어 있습니다.
CUDA runtime, cuDNN, cuBLAS, MPI의 설치 경로를 알맞게 지정하여 주십시오. 

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
스크립트의 host\_list를 클러스터 구성에 맞게 지정해주십시오.
스크립트의 dev\_list를 노드 구성에 맞게 지정해주십시오.
스크립트의 메인 루프문에 포함된 run\_test 함수에 알맞은 인자를 넣어주십시오.
[ run\_test {number of nodes} {number of devices} {model name} ]
실행 결과는 log/ 디렉터리에 저장됩니다.

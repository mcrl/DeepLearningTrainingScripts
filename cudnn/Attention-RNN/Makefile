TARGETS := attention-rnn-training attention-rnn-inference

NVCC := nvcc
CC := mpicc
CPP := mpic++

CFLAGS := -Wall -std=gnu99

SRC_PATH := ./src
OBJ_PATH := ./obj
LAYER_SRC_PATH := ./src/layers
OPTIMIZER_SRC_PATH := ./src/optimizers
LOSS_SRC_PATH := ./src/losses

INCLUDE_PATH := ./include

CC_INCLUDE := -I$(INCLUDE_PATH) -I/usr/local/cuda/include -I/home/daeyoung/cuda/include/ 
CC_LIBRARY := -lm -lrt

CC_CUDA_INCLUDE := -I/usr/local/cuda/include -I/home/daeyoung/cuda/include 
CC_CUDA_LIBRARY := -L/home/daeyoung/cuda/lib64 -L/usr/lib/x86_64-linux-gnu -L/usr/local/cuda/lib64/ -lcudart -lcublas -lcudnn -lcurand

CUDA_INCLUDE_PATH := /usr/local/cuda/include

SRCS := $(wildcard $(SRC_PATH)/*.c)
LAYER_SRCS := $(wildcard $(LAYER_SRC_PATH)/*.c)
OPTIMIZER_SRCS := $(wildcard $(OPTIMIZER_SRC_PATH)/*.c)
LOSS_SRCS := $(wildcard $(LOSS_SRC_PATH)/*.c)
CU_SRCS := $(wildcard $(SRC_PATH)/*.cu)
CPP_SRCS := $(wildcard $(SRC_PATH)/*.cpp $(LAYER_SRC_PATH)/*.cpp)

OBJS := $(patsubst $(SRC_PATH)/%.c,$(OBJ_PATH)/%.o,$(SRCS))
LAYER_OBJS := $(patsubst $(LAYER_SRC_PATH)/%.c,$(OBJ_PATH)/%.o,$(LAYER_SRCS))
OPTIMIZER_OBJS := $(patsubst $(OPTIMIZER_SRC_PATH)/%.c,$(OBJ_PATH)/%.o,$(OPTIMIZER_SRCS))
LOSS_OBJS := $(patsubst $(LOSS_SRC_PATH)/%.c,$(OBJ_PATH)/%.o,$(LOSS_SRCS))
CU_OBJS := $(patsubst $(SRC_PATH)/%.cu,$(OBJ_PATH)/%.o,$(CU_SRCS))
CPP_OBJS := $(patsubst $(SRC_PATH)/%.cpp,$(OBJ_PATH)/%.o,$(CPP_SRCS))


$(OBJ_PATH)/%.o: $(LAYER_SRC_PATH)/%.c
	$(CC) -c -o $@ $^ $(CC_INCLUDE) $(CC_CUDA_INCLUDE) $(CFLAGS) -O3

$(OBJ_PATH)/%.o: $(OPTIMIZER_SRC_PATH)/%.c
	$(CC) -c -o $@ $^ $(CC_INCLUDE) $(CC_CUDA_INCLUDE) $(CFLAGS) -O3

$(OBJ_PATH)/%.o: $(LOSS_SRC_PATH)/%.c
	$(CC) -c -o $@ $^ $(CC_INCLUDE) $(CC_CUDA_INCLUDE) $(CFLAGS) -O3

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.c
	$(CC) -c -o $@ $^ $(CC_INCLUDE) $(CC_CUDA_INCLUDE) $(CFLAGS) -O3

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cpp
	$(CPP) -std=c++11 -c -o $@ $^ $(CC_INCLUDE) -O3

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cu
	$(NVCC) -c -o $@ $^ $(CC_INCLUDE) -O3 -D_FORCE_INLINES


INCLUDES := $(CC_INCLUDE) $(CC_CUDA_INCLUDE) $(CC_LIBRARY) $(CC_CUDA_LIBRARY)
LIBS := -lstdc++ -lnccl
TRAINING_OBJS := $(filter-out $(OBJ_PATH)/inference.o, $(OBJS)) $(CPP_OBJS) $(CU_OBJS) $(LAYER_OBJS) $(OPTIMIZER_OBJS) $(LOSS_OBJS)
INFERENCE_OBJS := $(filter-out $(OBJ_PATH)/train.o, $(OBJS)) $(CPP_OBJS) $(CU_OBJS) $(LAYER_OBJS) $(OPTIMIZER_OBJS) $(LOSS_OBJS)


all: $(TARGETS)

attention-rnn-training: $(TRAINING_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDES) $(LIBS)

attention-rnn-inference: $(INFERENCE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDES) $(LIBS)

clean :
	rm $(TARGETS) | true
	rm $(OBJ_PATH)/* | true

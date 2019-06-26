CNN := vgg resnet
#CNN := vgg resnet densenet inception

NVCC := nvcc
CC := gcc
CXX := g++

SRC_PATH := ./src
OBJ_PATH := ./obj
INCLUDE_PATH := ./include

CC_INCLUDE := -I$(INCLUDE_PATH)
CC_LIBRARY := -lm

CC_CUDA_INCLUDE := -I/usr/include -I/usr/local/cuda/include -I/usr/local/cuda/cudnn-7.4.2/include
CC_CUDA_LIBRARY := -L/usr/lib/x86_64-linux-gnu/ -L/usr/lib64 -L/usr/local/cuda/lib64/ -L/usr/local/cuda/cudnn-7.4.2/lib64 -lcudnn -lcublas -lcudart -lnccl

CUDA_INCLUDE_PATH := -I/usr/local/cuda/include -I/usr/local/cuda/cudnn-7.4.2/include

SRCS := $(SRC_PATH)/main.c $(SRC_PATH)/input.c $(SRC_PATH)/conv.c $(SRC_PATH)/fc.c $(SRC_PATH)/elt.c $(SRC_PATH)/bn.c $(SRC_PATH)/pool.c $(SRC_PATH)/act.c $(SRC_PATH)/softmax.c $(SRC_PATH)/branch.c $(SRC_PATH)/bias.c $(SRC_PATH)/concat.c $(SRC_PATH)/memory.c $(SRC_PATH)/list.c
CU_SRCS := $(wildcard $(SRC_PATH)/*.cu)
CPP_SRCS := $(wildcard $(SRC_PATH)/*.cpp)

OBJS := $(patsubst $(SRC_PATH)/%.c,$(OBJ_PATH)/%.o,$(SRCS))
PURE_OBJS := $(filter_out obj/vgg.o obj/resnet.o obj/densenet.o obj/inception.o,$(OBJS))
CU_OBJS := $(patsubst $(SRC_PATH)/%.cu,$(OBJ_PATH)/%.o,$(CU_SRCS))
CPP_OBJS := $(patsubst $(SRC_PATH)/%.cpp,$(OBJ_PATH)/%.o,$(CPP_SRCS))

DEFINES :=
#DEFINES += -DTIME_LAYER
#DEFINES += -DPRINT_LOSS

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.c
	$(CC) -c -o $@ $^ $(CC_INCLUDE) $(CC_CUDA_INCLUDE) $(CFLAGS) $(DEFINES) -D_POSIX_C_SOURCE=199309L -std=c99 -O3

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cpp
	$(CXX) -c -o $@ $^ $(CC_INCLUDE) $(DEFINES) -std=c++11 -O3

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cu
	$(NVCC) -c -o $@ $^ $(CC_INCLUDE) $(CC_CUDA_INCLUDE) $(CFLAGS) $(DEFINES) -D_FORCE_INLINES -O3

.PHONY : all clean

all: vgg resnet
#all: vgg resnet densenet inception

vgg : obj/vgg.o $(OBJS) $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $@ $^ $(CC_INCLUDE) $(CC_CUDA_INCLUDE) $(CC_LIBRARY) $(CC_CUDA_LIBRARY) $(CFLAGS) -O3

resnet : obj/resnet.o $(OBJS) $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $@ $^ $(CC_INCLUDE) $(CC_CUDA_INCLUDE) $(CC_LIBRARY) $(CC_CUDA_LIBRARY) $(CFLAGS) -O3

densenet : obj/densenet.o $(OBJS) $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $@ $^ $(CC_INCLUDE) $(CC_CUDA_INCLUDE) $(CC_LIBRARY) $(CC_CUDA_LIBRARY) $(CFLAGS) -O3

inception : obj/inception.o $(OBJS) $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $@ $^ $(CC_INCLUDE) $(CC_CUDA_INCLUDE) $(CC_LIBRARY) $(CC_CUDA_LIBRARY) $(CFLAGS) -O3

clean :
	rm -f $(OBJ_PATH)/* | true
	rm -f $(CNN) | true

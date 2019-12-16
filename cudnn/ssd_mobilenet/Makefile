TARGET := ssd_mobilenet
CXX := g++
CXXFLAGS := -std=c++11 -O3
NVCC := nvcc

SRC_PATH := .
OBJ_PATH := ./obj
INCLUDE_PATH := .
COMMON_SRC_PATH := $(realpath ./common/src)
COMMON_INC_PATH := ./common/inc

CC_INCLUDE := -I$(INCLUDE_PATH) -I$(COMMON_INC_PATH)

CUDA_INCLUDE := -I/usr/local/cuda/cudnn-7.4.2/include -I/usr/local/cuda/include
CUDA_LIB := -L/usr/local/cuda/lib64 -L/usr/local/cuda/cudnn-7.4.2/lib64 -lcudnn -lcudart

CU_SRCS := $(wildcard $(SRC_PATH)/*.cu)
CPP_SRCS := $(wildcard $(SRC_PATH)/*.cpp)
COMMON_CPP_SRCS  := $(wildcard $(COMMON_SRC_PATH)/*.cpp)
COMMON_CU_SRCS  := $(wildcard $(COMMON_SRC_PATH)/*.cu)

CU_OBJS := $(patsubst $(SRC_PATH)/%.cu,$(OBJ_PATH)/%.o,$(CU_SRCS))
CPP_OBJS := $(patsubst $(SRC_PATH)/%.cpp,$(OBJ_PATH)/%.o,$(CPP_SRCS))
COMMON_CPP_OBJS := $(patsubst $(COMMON_SRC_PATH)/%.cpp,$(OBJ_PATH)/%.o,$(COMMON_CPP_SRCS))
COMMON_CU_OBJS := $(patsubst $(COMMON_SRC_PATH)/%.cu,$(OBJ_PATH)/%.o,$(COMMON_CU_SRCS))


$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cu
	$(NVCC) -c -o $@ $^ $(CC_INCLUDE) $(CUDA_INCLUDE) $(CFLAGS) $(CUFLAGS)

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cpp
	$(CXX) -c -o $@ $^ $(CC_INCLUDE) $(CUDA_INCLUDE) $(CXXFLAGS)

$(OBJ_PATH)/%.o: $(COMMON_SRC_PATH)/%.cpp
	$(CXX) -c -o $@ $^ $(CC_INCLUDE) $(CUDA_INCLUDE) $(CXXFLAGS)

$(OBJ_PATH)/%.o: $(COMMON_SRC_PATH)/%.cu
	$(NVCC) -c -o $@ $^ $(CC_INCLUDE) $(CUDA_INCLUDE) $(CXXFLAGS)

all : $(OBJS) $(CPP_OBJS) $(CU_OBJS) $(COMMON_CPP_OBJS) $(COMMON_CU_OBJS)
	$(CXX) -o $(TARGET) $^ $(CC_INCLUDE) $(CUDA_INCLUDE) $(CUDA_LIB) $(CFLAGS)

.phony : clean
clean :
	rm -rf $(TARGET) | true
	rm -rf $(OBJ_PATH)/* | true
	rm -rf ./data/results.txt

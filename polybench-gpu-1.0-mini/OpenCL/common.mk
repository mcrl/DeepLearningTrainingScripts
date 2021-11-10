OpenCL_SDK=/home/scott/NVIDIA_GPU_Computing_SDK
INCLUDE=-I${OpenCL_SDK}/OpenCL/common/inc
LIBPATH=-L${OpenCL_SDK}/OpenCL/common/lib -L${OpenCL_SDK}/shared/lib
LIB=-lOpenCL -lm
BENCH_FLAGS=-DWITH_SOURCE='"kernel0.cl"'
BENCH_FLAGS=-DWITH_BINARY='"kernel0.cl.aocx"'
all:
	gcc -O3 ${INCLUDE} ${LIBPATH} ${LIB} ${CFILES} -o ${EXECUTABLE} ${BENCH_FLAGS}

clean:
	rm -f *~ *.exe

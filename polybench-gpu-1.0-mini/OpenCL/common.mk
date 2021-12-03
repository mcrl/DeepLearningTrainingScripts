# GPU
#BENCH_FLAGS=-DWITH_SOURCE='"kernel0.cl"'

# Intel FPGA
#COMPILE_FLAGS=$(shell aocl compile-config)
#LINK_FLAGS=$(shell aocl link-config)
COMPILE_FLAGS=-I/opt/intelFPGA_pro/quartus_19.2.0b57/hld/host/include
LINK_FLAGS=-L/opt/intelFPGA_pro/quartus_19.2.0b57/hld/host/linux64/lib -lOpenCL -lm
#BENCH_FLAGS=-DINTEL -DWITH_BINARY='"kernel0.aocx"'

# soff
#COMPILE_FLAGS=-I/usr/local/cuda/include
#LINK_FLAGS=-L/usr/local/cuda/lib64 -lOpenCL
BENCH_FLAGS=-DSOFF -DWITH_BINARY='"kernel0.cl.sfb"'

BENCH_FLAGS+=-DMEASURE -DMEASURE_TIME_THRESHOLD=30 -DMEASURE_IDLE_WATT=355

all:
	gcc -O3 ${CFILES} -o ${EXECUTABLE} ${BENCH_FLAGS} ${COMPILE_FLAGS} ${LINK_FLAGS}

clean:
	rm -f *~ *.exe

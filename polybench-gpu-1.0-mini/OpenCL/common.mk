# Intel FPGA
COMPILE_FLAGS=-I/opt/intelFPGA_pro/quartus_19.2.0b57/hld/host/include
LINK_FLAGS=-L/opt/intelFPGA_pro/quartus_19.2.0b57/hld/host/linux64/lib -lOpenCL -lm
#BENCH_FLAGS=-DINTEL -DWITH_BINARY='"kernel0.aocx"'

# soff
BENCH_FLAGS=-DSOFF -DWITH_BINARY='"kernel0.cl.sfb"'

BENCH_FLAGS+=-DMEASURE -DMEASURE_TIME_THRESHOLD=30 -DMEASURE_IDLE_WATT=355

all:
	gcc -O3 ${CFILES} -o ${EXECUTABLE} ${BENCH_FLAGS} ${COMPILE_FLAGS} ${LINK_FLAGS}

clean:
	rm -f *~ *.exe

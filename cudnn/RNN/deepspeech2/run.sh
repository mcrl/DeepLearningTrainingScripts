mpirun -x LD_LIBRARY_PATH -npernode ${1} --hostfile hosts -H ${2} ./deepspeech.cudnn ${1} 

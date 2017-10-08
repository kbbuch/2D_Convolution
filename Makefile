# Location of the CUDA Toolkit
CUDA_PATH       := /usr/local/cuda/bin
NVCC := $(CUDA_PATH)/nvcc
CCFLAGS := -O3 -Wno-deprecated-gpu-targets -lineinfo

build: 2dconvol

2dconvol.o:2dconvol.cu
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

2dconvol: 2dconvol.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -Wno-deprecated-gpu-targets -lineinfo -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./2dconvol

clean:
	rm -f 2dconvol *.o

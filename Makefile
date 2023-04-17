MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
ROOT_DIR := $(patsubst %/,%,$(dir $(MKFILE_PATH)))

GPP:= /usr/bin/g++
ifeq ($(CUDA_HOME),)
	CUDA_HOME:= $(shell which nvcc | rev | cut -d'/' -f3- | rev)
endif

ifndef CUDA_VERSION
$(warning WARNING: CUDA_VERSION not set. Call make with CUDA string, for example: make cuda11x CUDA_VERSION=115 or make cpuonly CUDA_VERSION=CPU)
CUDA_VERSION:=
endif


NVCC := $(CUDA_HOME)/bin/nvcc

###########################################

CSRC := $(ROOT_DIR)/csrc
BUILD_DIR:= $(ROOT_DIR)/build

FILES_CUDA := $(CSRC)/ops.cu $(CSRC)/kernels.cu
# FILES_HIP := $(CSRC)/ops.cu $(CSRC)/kernels.cu
FILES_CPP := $(CSRC)/common.cpp $(CSRC)/cpu_ops.cpp $(CSRC)/pythonInterface.c

INCLUDE :=  -I $(CUDA_HOME)/include -I $(ROOT_DIR)/csrc -I $(CONDA_PREFIX)/include -I $(ROOT_DIR)/include
LIB := -L $(CUDA_HOME)/lib64 -lcudart -lcublas -lcublasLt -lcurand -lcusparse -L $(CONDA_PREFIX)/lib

# NVIDIA NVCC compilation flags
COMPUTE_CAPABILITY := -gencode arch=compute_35,code=sm_35 # Kepler 
COMPUTE_CAPABILITY += -gencode arch=compute_37,code=sm_37 # Kepler 
COMPUTE_CAPABILITY += -gencode arch=compute_50,code=sm_50 # Maxwell
COMPUTE_CAPABILITY += -gencode arch=compute_52,code=sm_52 # Maxwell
COMPUTE_CAPABILITY += -gencode arch=compute_60,code=sm_60 # Pascal
COMPUTE_CAPABILITY += -gencode arch=compute_61,code=sm_61 # Pascal
COMPUTE_CAPABILITY += -gencode arch=compute_70,code=sm_70 # Volta
COMPUTE_CAPABILITY += -gencode arch=compute_72,code=sm_72 # Volta 

# CUDA 9.2 supports CC 3.0, but CUDA >= 11.0 does not
CC_CUDA92 := -gencode arch=compute_30,code=sm_30

# Later versions of CUDA support the new architectures
CC_CUDA10x := -gencode arch=compute_30,code=sm_30
CC_CUDA10x += -gencode arch=compute_75,code=sm_75

CC_CUDA110 := -gencode arch=compute_75,code=sm_75
CC_CUDA110 += -gencode arch=compute_80,code=sm_80

# CC_CUDA11x := -gencode arch=compute_52,code=sm_52
CC_CUDA11x := -gencode arch=compute_75,code=sm_75
CC_CUDA11x += -gencode arch=compute_80,code=sm_80
CC_CUDA11x += -gencode arch=compute_86,code=sm_86

CC_cublasLt110 := -gencode arch=compute_75,code=sm_75
CC_cublasLt110 += -gencode arch=compute_80,code=sm_80

# CC_cublasLt111 := -gencode arch=compute_52,code=sm_52
CC_cublasLt111 := -gencode arch=compute_75,code=sm_75
CC_cublasLt111 += -gencode arch=compute_80,code=sm_80
CC_cublasLt111 += -gencode arch=compute_86,code=sm_86


all: $(BUILD_DIR) env
	$(NVCC) $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES_CUDA) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR) 
	$(NVCC) $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o -o $(BUILD_DIR)/link.o 
	$(GPP) -std=c++14 -DBUILD_CUDA -shared -fPIC $(INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./bitsandbytes/libbitsandbytes_cuda$(CUDA_VERSION).so $(LIB)

cuda92: $(BUILD_DIR) env
	$(NVCC) $(COMPUTE_CAPABILITY) $(CC_CUDA92) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES_CUDA) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR) -D NO_CUBLASLT
	$(NVCC) $(COMPUTE_CAPABILITY) $(CC_CUDA92) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o -o $(BUILD_DIR)/link.o 
	$(GPP) -std=c++14 -DBUILD_CUDA -shared -fPIC $(INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./bitsandbytes/libbitsandbytes_cuda$(CUDA_VERSION)_nocublaslt.so $(LIB)

cuda10x_nomatmul: $(BUILD_DIR) env
	$(NVCC) $(COMPUTE_CAPABILITY) $(CC_CUDA10x) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES_CUDA) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR) -D NO_CUBLASLT
	$(NVCC) $(COMPUTE_CAPABILITY) $(CC_CUDA10x) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o -o $(BUILD_DIR)/link.o 
	$(GPP) -std=c++14 -DBUILD_CUDA -shared -fPIC $(INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./bitsandbytes/libbitsandbytes_cuda$(CUDA_VERSION)_nocublaslt.so $(LIB)

cuda110_nomatmul: $(BUILD_DIR) env
	$(NVCC) $(COMPUTE_CAPABILITY) $(CC_CUDA110) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES_CUDA) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR) -D NO_CUBLASLT
	$(NVCC) $(COMPUTE_CAPABILITY) $(CC_CUDA110) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o -o $(BUILD_DIR)/link.o 
	$(GPP) -std=c++14 -DBUILD_CUDA -shared -fPIC $(INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./bitsandbytes/libbitsandbytes_cuda$(CUDA_VERSION)_nocublaslt.so $(LIB)

cuda11x_nomatmul: $(BUILD_DIR) env
	$(NVCC) $(COMPUTE_CAPABILITY) $(CC_CUDA11x) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES_CUDA) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR) -D NO_CUBLASLT
	$(NVCC) $(COMPUTE_CAPABILITY) $(CC_CUDA11x) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o -o $(BUILD_DIR)/link.o 
	$(GPP) -std=c++14 -DBUILD_CUDA -shared -fPIC $(INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./bitsandbytes/libbitsandbytes_cuda$(CUDA_VERSION)_nocublaslt.so $(LIB)

cuda110: $(BUILD_DIR) env
	$(NVCC) $(CC_cublasLt110) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES_CUDA) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR)
	$(NVCC) $(CC_cublasLt110) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o -o $(BUILD_DIR)/link.o 
	$(GPP) -std=c++14 -DBUILD_CUDA -shared -fPIC $(INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./bitsandbytes/libbitsandbytes_cuda$(CUDA_VERSION).so $(LIB)

cuda11x: $(BUILD_DIR) env
	$(NVCC) $(CC_cublasLt111) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES_CUDA) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR)
	$(NVCC) $(CC_cublasLt111) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o -o $(BUILD_DIR)/link.o 
	$(GPP) -std=c++14 -DBUILD_CUDA -shared -fPIC $(INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./bitsandbytes/libbitsandbytes_cuda$(CUDA_VERSION).so $(LIB)

cpuonly: $(BUILD_DIR) env
	$(GPP) -std=c++14 -shared -fPIC -I $(ROOT_DIR)/csrc -I $(ROOT_DIR)/include $(FILES_CPP) -o ./bitsandbytes/libbitsandbytes_cpu.so


HIP_INCLUDE := -I $(ROOT_DIR)/csrc -I $(ROOT_DIR)/include 
# -I /opt/rocm/hipcub/include
HIP_LIB := -L/opt/rocm/lib -L/opt/rocm/llvm/bin/../lib/clang/15.0.0/lib/linux -L/usr/lib/gcc/x86_64-linux-gnu/11 -L/usr/lib/gcc/x86_64-linux-gnu/11/../../../../lib64 -L/lib/x86_64-linux-gnu -L/lib/../lib64 -L/usr/lib/x86_64-linux-gnu -L/usr/lib/../lib64 -L/lib -L/usr/lib -lgcc_s -lgcc -lpthread -lm -lrt -lamdhip64 -lhipblas -lhipsparse -lclang_rt.builtins-x86_64 -lstdc++ -lm -lgcc_s -lgcc -lc -lgcc_s -lgcc

hip: $(BUILD_DIR)
	/usr/bin/hipcc -std=c++14 -c -fPIC --amdgpu-target=gfx1030 $(HIP_INCLUDE) -o $(BUILD_DIR)/ops.o -D NO_CUBLASLT $(CSRC)/ops.cu
	/usr/bin/hipcc -std=c++14 -c -fPIC --amdgpu-target=gfx1030 $(HIP_INCLUDE) -o $(BUILD_DIR)/kernels.o -D NO_CUBLASLT $(CSRC)/kernels.cu
	# /usr/bin/hipcc -fPIC -static $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o -o $(BUILD_DIR)/link.so 
	$(GPP) -std=c++14 -D__HIP_PLATFORM_AMD__ -DBUILD_CUDA -shared -fPIC -I /opt/rocm/include $(HIP_INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(FILES_CPP) $(HIP_LIB) -o ./bitsandbytes/libbitsandbytes_hip_nocublaslt.so 

env:
	@echo "ENVIRONMENT"
	@echo "============================"
	@echo "CUDA_VERSION: $(CUDA_VERSION)"
	@echo "============================"
	@echo "NVCC path: $(NVCC)"
	@echo "GPP path: $(GPP) VERSION: `$(GPP) --version | head -n 1`"
	@echo "CUDA_HOME: $(CUDA_HOME)"
	@echo "CONDA_PREFIX: $(CONDA_PREFIX)"
	@echo "PATH: $(PATH)"
	@echo "LD_LIBRARY_PATH: $(LD_LIBRARY_PATH)"
	@echo "============================"

$(BUILD_DIR):
	mkdir -p build
	mkdir -p dependencies

# $(ROOT_DIR)/dependencies/cub:
# 	git clone https://github.com/NVlabs/cub $(ROOT_DIR)/dependencies/cub
# 	cd dependencies/cub; git checkout 1.11.0

clean:
	rm build/* 

cleaneggs:
	rm -rf *.egg*

cleanlibs:
	rm ./bitsandbytes/libbitsandbytes*.so

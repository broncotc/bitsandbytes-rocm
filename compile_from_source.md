# Compiling from source
This has been ammended from the original source with details on how to compile for ROCm 5.5, and the 7900XT series. Ensure all other version of bitsandbytes are uninstalled before continuing.

I think this may work on the 6000 series too, but you may need to ammend the amdgpu-target from gfx1100 to the specific gfx for your card on Makefile lines 246 & 247:

'''
	/opt/rocm/bin/hipcc -std=c++14 -c -fPIC --amdgpu-target=gfx1100 $(HIP_INCLUDE) -o $(BUILD_DIR)/ops.o -D NO_CUBLASLT $(CSRC)/ops.cu
	/opt/rocm/bin/hipcc -std=c++14 -c -fPIC --amdgpu-target=gfx1100 $(HIP_INCLUDE) -o $(BUILD_DIR)/kernels.o -D NO_CUBLASLT $(CSRC)/kernels.cu
'''

There is probably a way to do this really cleanly, but i'm extreemly new to all of this!

You will need the ROCm hiplibsdk library at a minimum... probably.

```
git clone https://github.com/andrewcharlwood/bitsandbytes-rocm-7900XT/
cd bitsandbytes-rocm-7900XT
make hip
python setup.py install
```


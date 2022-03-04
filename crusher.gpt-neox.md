# Setup GPT-NeoX-20B Env

## Create Conda env
```
// install conda3 & initialize first

conda create -y -p /ccs/home/hsyoo/crusher_hackathon/conda_env python=3.8
conda activate /ccs/home/hsyoo/crusher_hackathon/conda_env
```

## Install PyTorch (rocm4.5 compatible pytorch provided by HPE)
```
pip install torch-1.11.0a0+git3e9e580-cp38-cp38-linux_x86_64.whl
```

## Install Apex (wheel provided by HPE)
```
pip install apex-0.1-cp38-cp38-linux_x86_64.whl
```

## Dependency install
### env
```
module load rocm/4.5.2 gcc/11.2.0

source /ccs/home/hsyoo/crusher_conda.sh
conda activate /gpfs/alpine/med106/proj-shared/hsyoo/Crusher/GPTNeoX20B/conda_env

export TORCH_EXTENSIONS_DIR=/ccs/home/hsyoo/crusher_neox/pytorch_extensions/
export MAX_JOBS=64
export HCC_AMDGPU_TARGET=gfx90a

export LD_PRELOAD="/opt/cray/pe/gcc/11.2.0/snos/lib64/libstdc++.so.6 /gpfs/alpine/world-shared/bip214/rocm_smi_lib/build/rocm_smi/librocm_smi64.so"
export LD_PRELOAD="${LD_PRELOAD} ${CRAY_MPICH_ROOTDIR}/gtl/lib/libmpi_gtl_hsa.so"
```

###  DeeperSpeed
```
git clone https://github.com/EleutherAI/DeeperSpeed.git && cd DeeperSpeed
pip install -r requirements/requirements.txt
./install.sh
cd ..
```

### GPT-NeoX
```
git clone https://github.com/EleutherAI/gpt-neox.git && cd gpt-neox
// comment out deeperspeed and mpi4py in requirements/requirements.txt
pip install -r requirements/requirements.txt

// checkout changes at https://github.com/hyoo/gpt-neox/commit/fec19db0c21c141c621a771f49ea2a74301ae48f
// for fused kernels
```

# Testing setup
```
#!/bin/bash

#SBATCH -A MED106_crusher
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J GPT_neox
#SBATCH -o %x-%j.out
#SBATCH -p batch

set +x
module load rocm/4.5.2 gcc/11.2.0

source /ccs/home/hsyoo/crusher_conda.sh
conda activate /gpfs/alpine/med106/proj-shared/hsyoo/Crusher/GPTNeoX20B/conda_env

export TORCH_EXTENSIONS_DIR=/ccs/home/hsyoo/crusher_neox/pytorch_extensions/
export MAX_JOBS=64
export HCC_AMDGPU_TARGET=gfx90a

export LD_PRELOAD="/opt/cray/pe/gcc/11.2.0/snos/lib64/libstdc++.so.6 /gpfs/alpine/world-shared/bip214/rocm_smi_lib/build/rocm_smi/librocm_smi64.so"
export LD_PRELOAD="${LD_PRELOAD} ${CRAY_MPICH_ROOTDIR}/gtl/lib/libmpi_gtl_hsa.so"

# train
python deepy.py train.py -d configs small.yml local_setup.yml
```

# DeepSpeed env
The Deep(er)Speed reads `.deepspeed_env` in the home directory to setup environmental variables of the worker nodes. Here is an example.
```
CONDA_DEFAULT_ENV=/gpfs/alpine/med106/proj-shared/hsyoo/Crusher/GPTNeoX20B/conda_env
ROCM_PATH=/opt/rocm-4.5.2
HIP_PATH=/opt/rocm-4.5.2/hip
LLVM_PATH=/opt/rocm-4.5.2/llvm
HIP_CLANG_PATH=/opt/rocm-4.5.2/llvm/bin
HSA_PATH=/opt/rocm-4.5.2
ROCMINFO_PATH=/opt/rocm-4.5.2
DEVICE_LIB_PATH=/opt/rocm-4.5.2/amdgcn/bitcode
HIP_DEVICE_LIB_PATH=/opt/rocm-4.5.2/amdgcn/bitcode
HIP_PLATFORM=amd
HIP_COMPILER=clang
OLCF_ROCM_ROOT=/opt/rocm-4.5.2
LD_LIBRARY_PATH=/opt/cray/pe/gcc/11.2.0/snos/lib64:/opt/rocm-4.5.2/lib64:/opt/rocm-4.5.2/lib:/opt/cray/pe/papi/6.0.0.12/lib64:/opt/cray/libfabric/1.15.0.0/lib64
CMAKE_PREFIX_PATH=/opt/rocm-4.5.2/hip:/opt/rocm-4.5.2
PATH=/gpfs/alpine/med106/proj-shared/hsyoo/Crusher/GPTNeoX20B/conda_env/bin:/opt/cray/pe/gcc/11.2.0/bin:/opt/rocm-4.5.2/bin:/ccs/home/hsyoo/bin:/ccs/home/hsyoo/programs/java/bin:/ccs/home/hsyoo/programs/crusher_conda3/condabin:/opt/cray/pe/craype/2.7.13/bin:/opt/cray/pe/cce/13.0.0/binutils/cross/x86_64-aarch64/bin:/opt/cray/pe/perftools/21.12.0/bin:/opt/cray/pe/papi/6.0.0.12/bin:/opt/cray/libfabric/1.15.0.0/bin:/opt/clmgr/sbin:/opt/clmgr/bin:/opt/sgi/sbin:/opt/sgi/bin:/usr/local/bin:/usr/bin:/bin:/opt/bin:/opt/c3/bin:/usr/lib/mit/bin:/opt/puppetlabs/bin:/sbin:/opt/cray/pe/bin
TORCH_EXTENSIONS_DIR=/ccs/home/hsyoo/crusher_neox/pytorch_extensions/
MAX_JOBS=64
CC=gcc
CXX=g++
LD_PRELOAD="/opt/cray/pe/gcc/11.2.0/snos/lib64/libstdc++.so.6 /gpfs/alpine/world-shared/bip214/rocm_smi_lib/build/rocm_smi/librocm_smi64.so /opt/cray/pe/mpich/8.1.12/gtl/lib/libmpi_gtl_hsa.so"
PE_MPICH_GTL_LIBS_amd_gfx908=-lmpi_gtl_hsa
HCC_AMDGPU_TARGET=gfx90a
ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```
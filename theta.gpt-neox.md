# Setup GPT-NeoX-20B Env

## Create conda env at login node (thetagpusn1 or thetagpusn2)
```
module load conda/2021-11-30 nccl/nccl-v2.11.4-1_CUDA11.4

conda create -p /projects/CSC249ADOA01/hsyoo/gpt-neox/conda_env python=3.8
conda activate /projects/CSC249ADOA01/hsyoo/gpt-neox/conda_env
```

## install pytorch
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

## install apex
```
cd /projects/CSC249ADOA01/hsyoo/gpt-neox/
git clone https://github.com/NVIDIA/apex.git && cd apex
pip install -r requirements.txt
```

### login to compute node for GPU env
```
qsub -A CSC249ADOA01 -t 60 -n 1 -I -q single-gpu
cd /projects/CSC249ADOA01/hsyoo/gpt-neox/apex/
CUDA_HOME=/usr/local/cuda-11.3 pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
exit
cd ..
```

## install deeperspeed
```
git clone https://github.com/EleutherAI/DeeperSpeed.git && cd DeeperSpeed
pip install -r requirements/requirements.txt
./install.sh
cd ..
```

## install gpt-neox
```
git clone https://github.com/EleutherAI/gpt-neox.git && cd gpt-neox
// comment out deeperspeed and mpi4py in requirements/requirements.txt
pip install -r requirements/requirements.txt
```

# Testing setup in interactive session
```
qsub -A CSC249ADOA01 -t 60 -n 1 -I

module load conda/2021-11-30
module load openmpi/openmpi-4.1.1_ucx-1.11.2_gcc-9.3.0
module load nccl/nccl-v2.11.4-1_CUDA11.4

conda activate /projects/CSC249ADOA01/hsyoo/gpt-neox/conda_env

// modify data path in configs/local_setup.yml
python deepy.py train.py -d configs small.yml local_setup.yml

```
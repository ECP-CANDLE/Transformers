echo 'Creating conda environment...'
conda create -n neox-3.8 python=3.8
source ~/anaconda3/etc/profile.d/conda.sh
conda activate neox-3.8
echo 'Activated conda environment...'
echo $CONDA_PREFIX

echo 'Cloning PyTorch'
cd /raid/ogokdemir
git clone --recursive https://github.com/pytorch/pytorch.git

echo 'Setting CMAKE env variables...'

export CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc
export CUDA_HOME=/usr/local/cuda

echo 'Installing magma cuda 11-3...'
conda install -c pytorch magma-cuda113

echo 'Installing PyTorch dependencies.'
conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-“$(dirname $(which conda))/../“}
cd pytorch

echo 'Building PyTorch from source...'
python setup.py install

echo 'Cloning GPT Neox...'
cd ..
git clone https://github.com/EleutherAI/gpt-neox.git
cd gpt-neox/

echo 'Installing gpt-neox requirements...'
pip install -r requirements/requirements.txt

echo 'Installing fused kernels...'

python ./megatron/fused_kernels/setup.py install

echo 'Getting the prompt files...'
scp -r stevens@rbdgx1.cels.anl.gov:/raid/stevens/GPT-Neox-20B/gpt-neox/20B_check
points .
scp -r stevens@rbdgx1.cels.anl.gov:/raid/stevens/GPT-Neox-20B/gpt-neox/GPT-OUTPU
Ts .

echo 'Generating answers to prompts...'
./deepy.py generate.py ./configs/20B.yml -i GPT-OUTPUTs/inputs/Drug_names_prompt.txt -o Drug_names_prompt_output.txt



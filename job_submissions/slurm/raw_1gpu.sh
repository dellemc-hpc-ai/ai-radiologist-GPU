#!/bin/bash

mkdir -p outputs
mkdir -p saved_weights

#SBATCH -N 1 # No. of nodes needed
#SBATCH -n 1 
#SBATCH -t 15:00:00
#SBATCH -J tfrec_1gpu # job name
#SBATCH -o outputs/tfrec_1gpu-%J.o
#SBATCH -e outputs/tfrec_1gpu-%J.o 
#SBATCH -C c4140,m,32gb,v100 #change the features as per your requirement, use "gpulist" to see the features of all nodes
#SBATCH -p gpuq #partition name

module load cuda10.0/toolkit/10.0.130
module load gcc/7.2.0
source  activate docker_pip2 


export LD_LIBRARY_PATH=$HOME/cuda:$HOME/cuda/include:$HOME/cuda/lib64:$HOME/modules/openmpi-4.0.0-flags-ucx/bin:$HOME/modules/openmpi-4.0.0-flags-ucx/include:$LD_LIBRARY_PATH
export PATH=$HOME/cuda:$HOME/cuda/include:$HOME/cuda/lib64:$HOME/modules/openmpi-4.0.0-flags-ucx/bin:$HOME/modules/openmpi-4.0.0-flags-ucx/include:$PATH
export OMPI_MCA_btl_openib_allow_ib=1



mpirun -np 1 --map-by socket python chexnet_densenet_raw_images.py --batch_size=64  --epochs=10 --skip_eval=1 --write_weights=0
#horovodrun -np 1 python $HOME/models/chexnet/official_resnet_tf_1.12.0.py  --batch_size=128  --epochs=10 

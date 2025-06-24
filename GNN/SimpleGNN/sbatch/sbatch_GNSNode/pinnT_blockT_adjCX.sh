#!/bin/bash

#SBATCH --job-name=pinnT_blockT_adjCX
#SBATCH --output=./Job_out_GNSNode/pinnT_blockT_adjCX.out
#SBATCH --gres=gpu:a100:1               # Request 1 A40 GPU
#SBATCH --time=24:00:00                  # Request 24 hours of runtime
#SBATCH --ntasks=1                       # Run a single task
#SBATCH --cpus-per-task=16               # Request 16 CPU cores
#SBATCH --partition=a100                  # Use the A40 partition
#SBATCH --error=./Job_out_GNSNode/pinnT_blockT_adjCX.err

export PYTHONPATH=/home/hpc/iwi5/iwi5295h/GNN_load_flow:$PYTHONPATH
cd ..

# Run your application or script
srun python train_valid_test.py --RUNNAME pinnT_blockT_adjCX --PINN --BLOCK_DIAG --ADJ_MODE cplx --BATCH 16 --EPOCHS 10 --LR 0.001 --VAL_EVERY 1 --PARQUET ./data/212100_variations_4_8_16_32_bus_grid.parquet
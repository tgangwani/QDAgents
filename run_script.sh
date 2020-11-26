#!/usr/bin/env bash

set -x 

MPI_RANKS=12
mpirun -np $MPI_RANKS python main.py --env-name "SparseCheetah-v2" --config-file "default_config.yaml" --seed=$RANDOM

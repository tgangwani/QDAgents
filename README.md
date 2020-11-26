This repository contains code for our paper [Harnessing Distribution Ratio Estimators for Learning Agents with Quality and Diversity](https://arxiv.org/abs/2011.02614), published at the Conference on Robot Learning (CoRL), 2020.

The code heavily uses the RL machinery from [this awesome repository](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) with RL algorithms implemented in PyTorch. We also use some functionality from [OpenAI baselines](https://github.com/openai/baselines). Different policies of the ensemble run in separate MPI ranks and communicate using the MPI protocol. We additionally provide a *self-imitation* option for the policy gradient, as proposed in [Learning Self-Imitating Diverse Policies](https://github.com/tgangwani/selfImitationDiverse). 

The code was tested with the following packages:

* python 3.6.6
* pytorch 0.4.1
* gym  0.10.8
* mpi4py 3.0.0

## Running command
To run MuJoCo experiments, use the command below. Edit _default_config.yaml_ to change the hyperparameters.
```
mpirun -np $MPI_RANKS python main.py --env-name "SparseCheetah-v2" --config-file "default_config.yaml" --seed=$RANDOM
```
The "SparseCheetah-v2" environment is created by modifying the "HalfCheetah-v2" from OpenAI-Gym. Please see the paper for details.

## Credits
1. [ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)
2. [OpenAI baselines](https://github.com/openai/baselines)
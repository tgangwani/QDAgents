import sys
import random
from collections import deque
import numpy as np
from mpi4py import MPI
import torch

from qd_agents.utils.baselines_support import logger
from qd_agents.utils.common_utils import get_new_dir
from qd_agents.utils.arguments import get_args
from qd_agents.rl.rl_agent import RLAgent

def setup(args):
    comm = MPI.COMM_WORLD
    args.comm = comm
    args.comm_size = comm.Get_size()
    env_name = args.env_name.split("-")[0]

    # new directory for storing logs and data-dumps
    if args.rank == 0:
        mpi_logs_dir = get_new_dir(root='MPILOGS/'+env_name)
    else:
        mpi_logs_dir = None

    mpi_logs_dir = comm.bcast(mpi_logs_dir, root=0)
    mpi_logs_dir += 'rank%d/'%args.rank

    # set logging - rank0 writes to log+stdout, all others write to log
    log_fmts = [logger.make_output_format(format='log', ev_dir=mpi_logs_dir)]
    if args.rank == 0:
        log_fmts += [logger.make_output_format(format='stdout', ev_dir='/tmp/void')]
    logger.update_output_formats(log_fmts)

    args.seed = args.seed + 10000 * args.rank
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.mpi_logs_dir = mpi_logs_dir

def main(args):
    if args.rank == 0:
        print('++ QD-{}-{} :: MPI Info: [Comm-size {}, self-rank {}] ++'.format(
            args.dice_type if args.dre_type == 'dice' else 'nce', args.divergence, args.comm_size, args.rank))

    if args.dre_type == 'nce':
        from qd_agents.networks.nce_manager import NetworksManager
    elif args.dre_type == 'dice':
        from qd_agents.networks.dice_manager import NetworksManager
    else: raise ValueError("Unknown DRE type. Supported options: {nce, dice}")

    episode_returns = deque(maxlen=50)
    rl_agent = RLAgent(args)
    manager = NetworksManager(args, rl_agent)

    for j in range(args.num_iterations):

        # collect agent-environment interaction data
        rl_agent.collect_rollout_batch(episode_returns)

        # update self-imitation discriminator and the density-ratio networks
        kernel_vals, ep_ret_all_ranks = manager.update(np.average(list(episode_returns)[-10:]))

        # update actor-critic parameters with PPO
        value_loss, action_loss = rl_agent.update(kernel_vals, anneal_coef=(1-float(j)/args.num_iterations))

        if args.rank == 0:
            ep_ret_all_ranks = [round(x,2) for x in ep_ret_all_ranks]
            if args.debug_mode:
                kv = [round(x, 2) for x in kernel_vals.values()]
                print("<debug> Value-loss: {:.2f}, Action-loss: {:.2f}, Kernel-values:{}".format(value_loss, action_loss, kv))
            print("[{}/{}] Episodic-returns (all ranks): {}".format(j, args.num_iterations, ep_ret_all_ranks))
            sys.stdout.flush()

        args.comm.Barrier()

if __name__ == '__main__':
    args = get_args()
    args.num_processes = 1      # future work: code needs modifications to run more than 1 env. in each rank
    args.rank = MPI.COMM_WORLD.Get_rank()
    setup(args)
    print(vars(args))

    if not args.rank == 0:
        # redirect print statements of other ranks to null.
        f = open('/dev/null', 'w')
        sys.stdout = f

    args.comm.Barrier()
    main(args)

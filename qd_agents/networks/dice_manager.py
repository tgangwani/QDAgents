import importlib
import numpy as np
import torch
from qd_agents.networks.discriminator import Discriminator
from qd_agents.buffers.super_pq import SuperPQ
from qd_agents.utils.batch_fifo_dice import BatchFIFO
from qd_agents.networks.abstract_manager import AbstractManager

class NetworksManager(AbstractManager):
    def __init__(self, args, rl_agent):
        print("++ Networks Manager for DICE ++")

        self.comm = args.comm
        self.rank = args.rank
        self.debug_mode = args.debug_mode
        self.neighbors = [i for i in range(args.comm_size) if i != args.rank]

        self.obs_dim = rl_agent.obs_dim
        self.acs_dim = rl_agent.acs_dim
        self.rl_agent = rl_agent
        self.dice_type = args.dice_type

        init_states = self.rl_agent.sample_init_states(args.num_steps)
        self.m_batch_buffer = BatchFIFO(capacity=2)

        dre = importlib.import_module(dict(
                dual_dice='qd_agents.networks.dre.dd_density_ratio',
                gen_dice='qd_agents.networks.dre.gd_density_ratio',
                value_dice='qd_agents.networks.dre.vd_density_ratio'
                )[self.dice_type])

        if self.dice_type in ['dual_dice', 'gen_dice']:
            self.nb_batch_buffer = {i : BatchFIFO(capacity=2) for i in self.neighbors}
            self.dratios = {i : dre.DensityRatio(self.obs_dim, self.acs_dim, hidden_dim=100, init_states=init_states, args=args) for i in self.neighbors}
            self.inverse_dratios = {i : dre.DensityRatio(self.obs_dim, self.acs_dim, hidden_dim=100, init_states=init_states, args=args) for i in self.neighbors}
        elif self.dice_type == 'value_dice':
            self.nb_batch_buffer = {i : BatchFIFO(capacity=2, obs_dim=self.obs_dim) for i in self.neighbors}
            self.dratios = {i : dre.DensityRatio(self.obs_dim, self.acs_dim, hidden_dim=100, init_states=init_states, rl_agent=rl_agent, nb_buffer=None, args=args) for i in self.neighbors}
            self.inverse_dratios = {i : dre.DensityRatio(self.obs_dim, self.acs_dim, hidden_dim=100, init_states=init_states, rl_agent=rl_agent, nb_buffer=self.nb_batch_buffer[i], args=args) for i in self.neighbors}
        else: raise ValueError("Unknown DRE")

        for i in self.neighbors:
            self.inverse_dratios[i].mute_param_update()

        self.si_discriminator = Discriminator(self.obs_dim, self.acs_dim, hidden_dim=100, rank=args.rank)

        # high level wrapper around a class that can manage multiple priority queues (if needed)
        self.super_pq = SuperPQ(count=args.num_pqs, capacity=args.pq_capacity)

        pcnt = len(self.dratios[self.neighbors[0]].get_flat())
        self.nb_idr_param_buffer = {i : np.zeros(pcnt, dtype=np.float32) for i in self.neighbors}

        self.nb_obs_buffer = {i : np.zeros((args.num_steps + 1, self.obs_dim), dtype=np.float32) for i in self.neighbors}
        self.nb_acs_buffer = {i : np.zeros((args.num_steps, self.acs_dim), dtype=np.float32) for i in self.neighbors}
        self.nb_masks_buffer = {i : np.zeros((args.num_steps, 1), dtype=np.float32) for i in self.neighbors}

        self.kernel_vals = {i : 0 for i in self.neighbors}
        self.temperature = args.temperature
        self.optim_params = {k:args.__dict__[k] for k in ['si_grad_steps']}

    def update(self, ep_ret):

        completed_trajs = list(self.rl_agent.latest_trajs.values())[:-1]
        assert len(completed_trajs) > 0, "No completed trajectory. Consider increasing args.num_steps"

        # randomly select one of the pq-buffers, and add completed trajectories to it
        pqb = self.super_pq.random_select(ignore_empty=True)
        for traj in completed_trajs:
            pqb.add_traj({**traj, 'score':sum(traj['rewards'])})

        # pqb.add_traj() does a deepcopy, hence we can free some memory
        for i in list(self.rl_agent.latest_trajs.keys())[:-1]:
            del self.rl_agent.latest_trajs[i]

        # Update aux_rewards with values from the discriminator
        self.si_discriminator.predict_batch_rewards(self.rl_agent.rollouts)

        # Perform multiple updates of the discriminator classifier using rollouts and pq-buffer
        self.si_discriminator.update(self.super_pq.random_select(),
                self.rl_agent.rollouts, num_grad_steps=self.optim_params['si_grad_steps'])

        # send density-ratio params to neighbors
        dr_params = {}
        for i in self.neighbors:
            dr_params[i] = np.copy(self.dratios[i].get_flat().numpy())
            self.comm.Isend(dr_params[i], dest=i, tag=5)

        # receive inverted density-ratio params from neighbors
        for i in self.neighbors:
            self.comm.Recv(self.nb_idr_param_buffer[i], source=i, tag=5)
            self.inverse_dratios[i].set_from_flat(torch.from_numpy(self.nb_idr_param_buffer[i]))

        self.comm.Barrier()

        m_obs = self.rl_agent.rollouts.raw_obs.view(-1, self.obs_dim)
        m_acs = self.rl_agent.rollouts.actions.view(-1, self.acs_dim)
        m_masks = self.rl_agent.rollouts.masks[1:].view(-1, 1)   # note the indexing [1:], such that (s,a,mask=0) would mean the episode ended with this action
        m_obs = m_obs.numpy(); m_acs = m_acs.numpy(); m_masks = m_masks.numpy()
        self.m_batch_buffer.add(m_obs, m_acs, m_masks)

        # Update aux_rewards with values from the density-ratio networks
        self.dratios[i].predict_batch_rewards(i, self.rl_agent.rollouts, self.inverse_dratios[i], self.rl_agent.get_action)

        # send (s,a, mask) data to all neighbors
        for i in self.neighbors:
            self.comm.Isend(m_obs, dest=i, tag=33)
            self.comm.Isend(m_acs, dest=i, tag=55)
            self.comm.Isend(m_masks, dest=i, tag=11)

        # receive (s,a, mask) data from all neighbors
        for i in self.neighbors:
            self.comm.Recv(self.nb_obs_buffer[i], source=i, tag=33)
            self.comm.Recv(self.nb_acs_buffer[i], source=i, tag=55)
            self.comm.Recv(self.nb_masks_buffer[i], source=i, tag=11)
            self.nb_batch_buffer[i].add(self.nb_obs_buffer[i], self.nb_acs_buffer[i], self.nb_masks_buffer[i])

        losses = []
        dratios = []
        for i in self.neighbors:
            loss_val, dratio_val, neg_divergence = self.dratios[i].update(self.m_batch_buffer, self.nb_batch_buffer[i], self.rl_agent.get_action, self.inverse_dratios[i])
            losses.append(loss_val)
            dratios.append(dratio_val)
            neg_divergence /= self.temperature
            self.kernel_vals[i] = np.exp(neg_divergence)

        if self.debug_mode:
            print('<debug> Density Ratio stats: Avg:{0:.2f}, Max:{1:.2f}, Min:{2:.2f}, Avg-loss:{3:.2f}'.format(
                np.average(dratios), np.max(dratios), np.min(dratios), np.average(losses)))

        # sync!
        ep_ret_all_ranks = self.comm.gather(ep_ret, root=0)
        return self.kernel_vals, ep_ret_all_ranks

import torch
import numpy as np
from qd_agents.networks.discriminator import Discriminator
from qd_agents.networks.pdf import PdfNetwork
from qd_agents.buffers.super_pq import SuperPQ
from qd_agents.utils.batch_fifo_nce import BatchFIFO
from qd_agents.networks.abstract_manager import AbstractManager

class NetworksManager(AbstractManager):
    def __init__(self, args, rl_agent):
        print("++ Networks Manager for NCE ++")

        self.comm = args.comm
        self.rank = args.rank
        self.debug_mode = args.debug_mode
        self.neighbors = [i for i in range(args.comm_size) if i != args.rank]

        self.obs_dim = rl_agent.obs_dim
        self.acs_dim = rl_agent.acs_dim
        self.rl_agent = rl_agent

        self.si_discriminator = Discriminator(self.obs_dim, self.acs_dim, hidden_dim=100, rank=args.rank)
        self.pdfs = {i : PdfNetwork(self.obs_dim, self.acs_dim, hidden_dim=100, args=args) for i in range(args.comm_size)}

        for i in self.neighbors:
            self.pdfs[i].mute_param_update()

        # high level wrapper around a class that can manage multiple priority queues (if needed)
        self.super_pq = SuperPQ(count=args.num_pqs, capacity=args.pq_capacity)

        pcnt = len(self.pdfs[self.rank].get_flat())
        self.nb_pdf_param_buffer = {i : np.zeros(pcnt, dtype=np.float32) for i in self.neighbors}
        self.nb_obs_buffer = {i : np.zeros((args.num_steps, self.obs_dim), dtype=np.float32) for i in self.neighbors}
        self.nb_acs_buffer = {i : np.zeros((args.num_steps, self.acs_dim), dtype=np.float32) for i in self.neighbors}

        self.m_batch_buffer = BatchFIFO(capacity=2)
        self.nb_batch_buffer = {i : BatchFIFO(capacity=2) for i in self.neighbors}
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

        self.comm.Barrier()

        # send self-pdf parameters to all neighbors
        m_pdf_params = self.pdfs[self.rank].get_flat().detach().numpy()
        for i in self.neighbors:
            self.comm.Isend(m_pdf_params, dest=i, tag=11)

        # receive latest pdf parameters from neighbor ranks
        for i in self.neighbors:
            self.comm.Recv(self.nb_pdf_param_buffer[i], source=i, tag=11)
            self.pdfs[i].set_from_flat(torch.from_numpy(self.nb_pdf_param_buffer[i]))

        m_obs = self.rl_agent.rollouts.raw_obs[:-1].view(-1, self.obs_dim)
        m_acs = self.rl_agent.rollouts.actions.view(-1, self.acs_dim)
        m_pdf = self.pdfs[self.rank].infer(torch.cat([m_obs, m_acs], dim=1))
        m_obs = m_obs.numpy(); m_acs = m_acs.numpy()
        self.m_batch_buffer.add(m_obs, m_acs)

        # Update aux_rewards with values from the PDF networks
        for i in self.neighbors:
            self.pdfs[i].predict_batch_rewards(i, m_pdf, self.rl_agent.rollouts)

        # send (s,a) data to all neighbors
        for i in self.neighbors:
            self.comm.Isend(m_obs, dest=i, tag=33)
            self.comm.Isend(m_acs, dest=i, tag=55)

        # receive (s,a) data from all neighbors
        for i in self.neighbors:
            self.comm.Recv(self.nb_obs_buffer[i], source=i, tag=33)
            self.comm.Recv(self.nb_acs_buffer[i], source=i, tag=55)
            self.nb_batch_buffer[i].add(self.nb_obs_buffer[i], self.nb_acs_buffer[i])

        losses = []
        for i in self.neighbors:
            loss_val, neg_divergence = self.pdfs[self.rank].update(self.m_batch_buffer, self.nb_batch_buffer[i], self.pdfs, i)
            losses.append(loss_val)
            neg_divergence /= self.temperature
            self.kernel_vals[i] = np.exp(neg_divergence)

        if self.debug_mode:
            print('<debug> Pdf (unnormalized) stats: Avg:{0:.2f}, Max:{1:.2f}, Min:{2:.2f}, Avg-loss:{3:.2f}'.format(
                m_pdf.mean(0).item(), m_pdf.max(0)[0].item(), m_pdf.min(0)[0].item(), np.average(losses)))

        # sync!
        ep_ret_all_ranks = self.comm.gather(ep_ret, root=0)
        return self.kernel_vals, ep_ret_all_ranks

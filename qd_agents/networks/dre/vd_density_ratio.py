from itertools import count
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from qd_agents.utils.common_utils import obs_batch_normalize, RunningMeanStd, init, CustomMapDataset, get_dr_limits
from qd_agents.utils.distributions import DiagGaussian

EPS = 1e-5
DR_MIN, DR_MAX = get_dr_limits()
LOG_DR_MIN, LOG_DR_MAX = np.log(DR_MIN), np.log(DR_MAX)

class DensityRatio(nn.Module):
    _ids = count(0)

    def __init__(self, ob_dim, ac_dim, hidden_dim, init_states, rl_agent, nb_buffer, args):

        # only print on creation of the first instance
        if not next(self._ids):
            print("++ ValueDICE Density Ratio Estimator ++")

        super().__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        input_dim = ob_dim + ac_dim
        self.gamma = args.gamma
        self.rl_agent = rl_agent
        self.nb_buffer = nb_buffer
        self.init_states = init_states
        self.rank = args.rank
        self.divergence = args.divergence
        actv = nn.Tanh
        gain = nn.init.calculate_gain('tanh')

        init_ = lambda m, bias=0: init(m, nn.init.xavier_normal_, lambda x: nn.init.
                constant_(x, bias), gain=gain)

        self.q = nn.Sequential(
                init_(nn.Linear(input_dim, hidden_dim)), actv(),
                init_(nn.Linear(hidden_dim, hidden_dim)), actv(),
                init_(nn.Linear(hidden_dim, hidden_dim)), actv(),
                init_(nn.Linear(hidden_dim, 1)))

        # for value-dice, we don't get the density ratio directly from a neural network, but
        # instead need to perform some calculation. Using notation of our paper, zeta_{ij}
        # computation requires the policy pi_i, similarly, zeta_{ji} requires pi_j.
        # self.dr_policy is this policy needed for that computation. This is used in the
        # inverse_dratios instantiations.
        self.dr_policy = nn.Sequential(
                init_(nn.Linear(ob_dim, 64)), nn.Tanh(),
                init_(nn.Linear(64, 64)), nn.Tanh(),
                DiagGaussian(64, ac_dim))
        for p in self.dr_policy.parameters():
            p.requires_grad = False

        # TODO: Ideally, we should be creating a policy object of the same architecture as
        # used by the rl_class automatically, w/o hardcoded layers and sizes. So, the code above
        # needs to be cleaned up in the future. Currently we have this placeholder sanity check
        assert sum(p.numel() for p in self.dr_policy.parameters()) == \
                sum(p.numel() for p in self.rl_agent.get_actor_params())

        # these three parameters are only used for sharing the normalization data across MPI ranks
        self._nrml_mean = nn.Parameter(torch.zeros(input_dim), requires_grad=False)
        self._nrml_var = nn.Parameter(torch.zeros(input_dim), requires_grad=False)
        self._nrml_count = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.input_rms = RunningMeanStd(shape=input_dim)
        self.optimizer = torch.optim.Adam((p for p in self.parameters() if p.requires_grad), lr=5e-4, weight_decay=1e-3)

    def compute_grad_pen(self, sa, next_sa, lambda_):
        alpha = torch.rand(sa.size(0), 1)
        alpha = alpha.expand_as(sa)

        mixup_data = alpha * sa + (1 - alpha) * next_sa
        mixup_data.requires_grad = True

        out = self.q(mixup_data)
        ones = torch.ones(out.size())

        grad = autograd.grad(
            outputs=out,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def mute_param_update(self):
        for p in self.parameters():
            p.requires_grad = False

    def set_from_flat(self, vector):
        vector_to_parameters(vector, self.parameters())
        self.input_rms.mean = self._nrml_mean.data.numpy()
        self.input_rms.var = self._nrml_var.data.numpy()
        self.input_rms.count = self._nrml_count.data.numpy()

    def get_flat(self):
        # update actor data
        actor_params = parameters_to_vector(self.rl_agent.get_actor_params()).detach()
        vector_to_parameters(actor_params, self.dr_policy.parameters())
        # update normalization data
        self._nrml_mean.data.copy_(torch.tensor(self.input_rms.mean))
        self._nrml_var.data.copy_(torch.tensor(self.input_rms.var))
        self._nrml_count.data.copy_(torch.tensor(self.input_rms.count))
        return parameters_to_vector(self.parameters()).detach()

    def eval_neg_divergence(self, sa, next_sa, masks, m_buffer, nb_inverse_dr, m_policy_act_fn):

        with torch.no_grad():
            if self.divergence == 'js':
                ratio1 = self._log_ratio(sa, next_sa).exp()
                m_states, m_actions, m_next_states, m_masks = m_buffer.get_sample(nbatches=2)
                ratio2 = nb_inverse_dr.eval_ratio(m_states, m_actions, m_next_states)

                output = torch.log(1. / (1 + ratio1) + EPS) * masks + torch.log(1. / (1. + ratio2) + EPS) * m_masks
                neg_js = 0.5 * (-output.mean(0) - torch.log(torch.tensor(4.)))
                neg_js = torch.clamp(neg_js, max=0)   # clip the residual positive part (if any)
                return neg_js.item()

            if self.divergence == 'kls':
                m_states, m_actions, m_next_states, m_masks = m_buffer.get_sample(nbatches=2)
                m_next_actions = m_policy_act_fn(m_next_states, deterministic=False)
                m_sa = torch.cat([m_states, m_actions], dim=1)
                m_sa = obs_batch_normalize(m_sa, update_rms=False, rms_obj=self.input_rms)

                m_next_sa = torch.cat([m_next_states, m_next_actions], dim=1)
                m_next_sa = obs_batch_normalize(m_next_sa, update_rms=False, rms_obj=self.input_rms)

                ratio1 = self._log_ratio(m_sa, m_next_sa).exp()
                ratio2 = 1./self._log_ratio(sa, next_sa).exp()

                output = torch.log(ratio1 + EPS) * m_masks + torch.log(ratio2 + EPS)  * masks
                neg_kls = -output.mean(0)
                neg_kls = torch.clamp(neg_kls, max=0)   # clip the residual positive part (if any)
                return neg_kls.item()

            raise ValueError("Unknown divergence")

    def _log_ratio(self, sa, next_sa):
        return (self.q(sa) - self.gamma * self.q(next_sa)).clamp(min=LOG_DR_MIN, max=LOG_DR_MAX)

    def eval_ratio(self, states, actions, next_states, deterministic=False):
        self.eval()
        sa = torch.cat([states, actions], dim=1)
        # (approximation) normalize states using the rms-object from the corresponding nb-batch-buffer
        normalized_next_states = obs_batch_normalize(next_states, update_rms=False, rms_obj=self.nb_buffer.obs_rms)

        with torch.no_grad():
            next_actions_distr = self.dr_policy(normalized_next_states)
            next_actions = next_actions_distr.sample()[1] if not deterministic else next_actions_distr.mode()[1]
            next_sa = torch.cat([next_states, next_actions], dim=1)
            sa = obs_batch_normalize(sa, update_rms=False, rms_obj=self.input_rms)
            next_sa = obs_batch_normalize(next_sa, update_rms=False, rms_obj=self.input_rms)
            ratio = self._log_ratio(sa, next_sa).exp()

        return ratio

    def update(self, m_buffer, nb_buffer, m_policy_act_fn, nb_inverse_dr):
        self.train()

        with torch.no_grad():
            init_actions = m_policy_act_fn(self.init_states, deterministic=False)
        init_sa = torch.cat([self.init_states, init_actions], dim=1)
        init_sa = obs_batch_normalize(init_sa, update_rms=True, rms_obj=self.input_rms)

        nb_states, nb_actions, nb_next_states, nb_masks = nb_buffer.get_sample(nbatches=2)
        with torch.no_grad():
            next_actions = m_policy_act_fn(nb_next_states, deterministic=False)

        sa = torch.cat([nb_states, nb_actions], dim=1)
        next_sa = torch.cat([nb_next_states, next_actions], dim=1)
        sa = obs_batch_normalize(sa, update_rms=True, rms_obj=self.input_rms)
        next_sa = obs_batch_normalize(next_sa, update_rms=True, rms_obj=self.input_rms)

        if sa.size(0) > init_sa.size(0):
            assert sa.size(0) % init_sa.size(0) == 0
            repeat_fac = int(sa.size(0) / init_sa.size(0))
            init_sa = init_sa.repeat(repeat_fac, 1)

        dset = CustomMapDataset(init_sa, sa, next_sa, nb_masks)
        dset_generator = torch.utils.data.DataLoader(dset, batch_size=512, shuffle=True)

        loss_val = 0.
        n = 0.
        for data in dset_generator:
            init_sa_mb, sa_mb, next_sa_mb, masks_mb = data

            td_err_exp = self._log_ratio(sa_mb, next_sa_mb).exp()
            td_err_exp = (td_err_exp * masks_mb).mean(0)
            loss_1 = torch.log(td_err_exp + EPS)

            loss_2 = -1. * (1 - self.gamma) * self.q(init_sa_mb).mean(0)

            grad_penalty = self.compute_grad_pen(sa_mb, next_sa_mb, lambda_=1.)

            # value-dice loss is computed as per the equations in [https://arxiv.org/abs/1912.05032]
            loss = loss_1 + loss_2 + grad_penalty

            loss_val += loss.item()
            n += 1

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.)
            self.optimizer.step()

        neg_divergence = self.eval_neg_divergence(sa, next_sa, nb_masks, m_buffer, nb_inverse_dr, m_policy_act_fn)
        return loss_val/n, -42., neg_divergence     # -42 proxy for None

    def predict_batch_rewards(self, idx, rollouts, nb_inverse_dr, m_policy_act_fn):
        self.eval()
        assert idx != self.rank
        obs = rollouts.raw_obs[:-1].view(-1, self.ob_dim)
        acs = rollouts.actions.view(-1, self.ac_dim)
        next_obs = rollouts.raw_obs[1:].view(-1, self.ob_dim)
        sa = torch.cat([obs, acs], dim=1)

        with torch.no_grad():
            next_acs = m_policy_act_fn(next_obs, deterministic=True)
            next_sa = torch.cat([next_obs, next_acs], dim=1)
            sa = obs_batch_normalize(sa, update_rms=False, rms_obj=self.input_rms)
            next_sa = obs_batch_normalize(next_sa, update_rms=False, rms_obj=self.input_rms)

            if self.divergence == 'js':
                ratio = self._log_ratio(sa, next_sa).exp()
                rewards = -torch.log(1. / (1. + ratio) + EPS)

            elif self.divergence == 'kls':
                ratio = nb_inverse_dr.eval_ratio(obs, acs, next_obs, deterministic=True)
                rewards = -ratio - torch.log(ratio + EPS)

            else: raise ValueError("Unknown divergence")

            masks = rollouts.masks[1:].view(-1, 1)
            rewards = rewards * masks
            rewards = rewards.view(rollouts.num_steps, -1, 1)
            rollouts.aux_rewards[idx].copy_(rewards)

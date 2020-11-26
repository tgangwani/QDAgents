from itertools import count
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from qd_agents.utils.common_utils import obs_batch_normalize, RunningMeanStd, init, CustomMapDataset, get_dr_limits

EPS = 1e-5
DR_MIN, DR_MAX = get_dr_limits()

class DensityRatio(nn.Module):
    _ids = count(0)

    def __init__(self, ob_dim, ac_dim, hidden_dim, init_states, args):

        # only print on creation of the first instance
        if not next(self._ids):
            print("++ GenDICE Density Ratio Estimator ++")

        super().__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        input_dim = ob_dim + ac_dim
        self.init_states = init_states
        self.rank = args.rank
        self.gamma = args.gamma
        self.divergence = args.divergence
        actv = nn.Tanh
        gain = nn.init.calculate_gain('tanh')

        init_ = lambda m, bias=0: init(m, nn.init.xavier_normal_, lambda x: nn.init.
                constant_(x, bias), gain=gain)

        self.tau = nn.Sequential(
                init_(nn.Linear(input_dim, hidden_dim)), actv(),
                init_(nn.Linear(hidden_dim, hidden_dim)), actv(),
                init_(nn.Linear(hidden_dim, hidden_dim)), actv(),
                init_(nn.Linear(hidden_dim, 1)))

        self.f = nn.Sequential(
                init_(nn.Linear(input_dim, hidden_dim)), actv(),
                init_(nn.Linear(hidden_dim, hidden_dim)), actv(),
                init_(nn.Linear(hidden_dim, hidden_dim)), actv(),
                init_(nn.Linear(hidden_dim, 1)))

        self.u = nn.Parameter(-1.*torch.ones(1))

        # these three parameters are only used for sharing the normalization data across MPI ranks
        self._nrml_mean = nn.Parameter(torch.zeros(input_dim), requires_grad=False)
        self._nrml_var = nn.Parameter(torch.zeros(input_dim), requires_grad=False)
        self._nrml_count = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.input_rms = RunningMeanStd(shape=input_dim)
        self.optimizer = torch.optim.Adam((p for p in self.parameters() if p.requires_grad), lr=5e-4, weight_decay=1e-3)

    @staticmethod
    def _modifier(x):
        x = nn.functional.softplus(x, beta=1., threshold=1.).clamp(min=DR_MIN, max=DR_MAX)
        return x

    def eval_ratio(self, sa):
        self.eval()
        with torch.no_grad():
            sa = obs_batch_normalize(sa, update_rms=False, rms_obj=self.input_rms)
            ratio = self._modifier(self.tau(sa))
        return ratio

    def mute_param_update(self):
        for p in self.parameters():
            p.requires_grad = False

    def set_from_flat(self, vector):
        vector_to_parameters(vector, self.parameters())
        self.input_rms.mean = self._nrml_mean.data.numpy()
        self.input_rms.var = self._nrml_var.data.numpy()
        self.input_rms.count = self._nrml_count.data.numpy()

    def get_flat(self):
        self._nrml_mean.data.copy_(torch.tensor(self.input_rms.mean))
        self._nrml_var.data.copy_(torch.tensor(self.input_rms.var))
        self._nrml_count.data.copy_(torch.tensor(self.input_rms.count))
        return parameters_to_vector(self.parameters()).detach()

    def eval_neg_divergence(self, m_buffer, nb_buffer, nb_inverse_dr):
        m_states, m_actions, *_ = m_buffer.get_sample(nbatches=2)
        nb_states, nb_actions, *_ = nb_buffer.get_sample(nbatches=2)
        m_sa = torch.cat([m_states, m_actions], dim=1)
        nb_sa = torch.cat([nb_states, nb_actions], dim=1)

        with torch.no_grad():

            if self.divergence == 'js':
                ratio1 = self.eval_ratio(nb_sa)
                ratio2 = nb_inverse_dr.eval_ratio(m_sa)

                output = torch.log(1. / (1. + ratio2) + EPS) + torch.log(1. / (1 + ratio1) + EPS)
                neg_js = 0.5 * (-output.mean(0) - torch.log(torch.tensor(4.)))
                neg_js = torch.clamp(neg_js, max=0)   # clip the residual positive part (if any)
                return neg_js.item()

            if self.divergence == 'kls':
                ratio1 = self.eval_ratio(m_sa)
                ratio2 = 1./self.eval_ratio(nb_sa)

                output = torch.log(ratio1 + EPS) + torch.log(ratio2 + EPS)
                neg_kls = -output.mean(0)
                neg_kls = torch.clamp(neg_kls, max=0)   # clip the residual positive part (if any)
                return neg_kls.item()

            raise ValueError("Unknown divergence")

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
        dratio_val = 0.
        for data in dset_generator:
            init_sa_mb, sa_mb, next_sa_mb, masks_mb = data

            loss_1 = (1 - self.gamma) * self.f(init_sa_mb).mean(0)

            loss_2 = self.gamma * self._modifier(self.tau(sa_mb)) * self.f(next_sa_mb)
            loss_2 = (loss_2 * masks_mb).mean(0)

            loss_3 = - self._modifier(self.tau(sa_mb)) * self.f(sa_mb) * (1 + 0.25*self.f(sa_mb))
            loss_3 = loss_3.mean(0)

            loss_4 = self._modifier(self.tau(sa_mb)) * self.u - self.u - 0.5*self.u.pow(2)
            loss_4 = (1. * loss_4).mean(0)

            # gen-dice loss is computed as per the equations in [https://arxiv.org/abs/2002.09072]
            loss = loss_1 + loss_2 + loss_3 + loss_4

            loss_val += loss.item()
            n += 1

            self.optimizer.zero_grad()
            loss.backward()

            for p in self.f.parameters():
                p.grad.data = -p.grad.data
            self.u.grad.data = -self.u.grad.data

            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.)
            self.optimizer.step()

            with torch.no_grad():
                dratio_val += self._modifier(self.tau(sa_mb)).mean(0).item()

        neg_divergence = self.eval_neg_divergence(m_buffer, nb_buffer, nb_inverse_dr)
        return loss_val/n, dratio_val/n, neg_divergence

    def predict_batch_rewards(self, idx, rollouts, nb_inverse_dr, *_):
        assert idx != self.rank
        obs = rollouts.raw_obs[:-1].view(-1, self.ob_dim)
        acs = rollouts.actions.view(-1, self.ac_dim)
        sa = torch.cat([obs, acs], dim=1)

        if self.divergence == 'js':
            ratio = self.eval_ratio(sa)
            rewards = -torch.log(1. / (1. + ratio) + EPS)

        elif self.divergence == 'kls':
            inv_ratio = nb_inverse_dr.eval_ratio(sa)
            rewards = -inv_ratio - torch.log(inv_ratio + EPS)

        else: raise ValueError("Unknown divergence")

        rewards = rewards.view(rollouts.num_steps, -1, 1)
        rollouts.aux_rewards[idx].copy_(rewards)

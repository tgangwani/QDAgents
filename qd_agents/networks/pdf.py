from itertools import count
import torch
import torch.nn as nn
from torch import autograd
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from qd_agents.utils.common_utils import obs_batch_normalize, RunningMeanStd, init, CustomMapDataset, get_dr_limits

EPS = 1e-5
DR_MIN, DR_MAX = get_dr_limits()

class PdfNetwork(nn.Module):
    _ids = count(0)

    def __init__(self, ob_dim, ac_dim, hidden_dim, args):

        # only print on creation of the first instance
        if not next(self._ids):
            print("++ Density Ratio Estimation with PDF Networks ++")

        super().__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        input_dim = ob_dim + ac_dim
        self.rank = args.rank
        self.divergence = args.divergence
        actv = nn.Tanh
        gain = nn.init.calculate_gain('tanh')

        init_ = lambda m, bias=0: init(m, nn.init.xavier_normal_, lambda x: nn.init.
                constant_(x, bias), gain=gain)

        self.tower = nn.Sequential(
                init_(nn.Linear(input_dim, hidden_dim)), actv(),
                init_(nn.Linear(hidden_dim, hidden_dim)), actv(),
                init_(nn.Linear(hidden_dim, hidden_dim)), actv(),
                init_(nn.Linear(hidden_dim, 1, bias=False)))

        # log partition function
        self.logZ = nn.Parameter(torch.ones(1))

        # these three parameters are only used for sharing the normalization data across MPI ranks
        self._nrml_mean = nn.Parameter(torch.zeros(input_dim), requires_grad=False)
        self._nrml_var = nn.Parameter(torch.zeros(input_dim), requires_grad=False)
        self._nrml_count = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.input_rms = RunningMeanStd(shape=input_dim)
        self.optimizer = torch.optim.Adam((p for p in self.parameters() if p.requires_grad), lr=5e-4, weight_decay=1e-3)

    def _modifier(self, m):
        m = torch.clamp(m - self.logZ, min=-5., max=5.)
        return m.exp()

    @staticmethod
    def _dr(numer, denom):
        return torch.div(numer, denom).clamp(min=DR_MIN, max=DR_MAX)

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

    def compute_grad_pen(self, m_sa, nb_sa, lambda_):
        alpha = torch.rand(m_sa.size(0), 1)
        alpha = alpha.expand_as(m_sa)

        mixup_data = alpha * m_sa + (1 - alpha) * nb_sa
        mixup_data.requires_grad = True

        out = self._modifier(self.tower(mixup_data))
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

    def eval_neg_divergence(self, m_states, m_actions, nb_states, nb_actions, nb_pdf_net):

        with torch.no_grad():
            m_sa = torch.cat([m_states, m_actions], dim=1)
            nb_sa = torch.cat([nb_states, nb_actions], dim=1)

            nb_pdf_m_data = nb_pdf_net.infer(m_sa).detach()
            nb_pdf_nb_data = nb_pdf_net.infer(nb_sa).detach()
            normalized_sa = obs_batch_normalize(torch.cat([m_sa, nb_sa], dim=0), update_rms=False, rms_obj=self.input_rms)
            m_sa, nb_sa = torch.split(normalized_sa, [m_states.size(0), nb_states.size(0)], dim=0)

            m_pdf_m_data = self._modifier(self.tower(m_sa))
            m_pdf_nb_data = self._modifier(self.tower(nb_sa))

            if self.divergence == 'js':
                ratio1 = self._dr(nb_pdf_m_data, m_pdf_m_data)
                ratio2 = self._dr(m_pdf_nb_data, nb_pdf_nb_data)

                output = torch.log(1. / (1. + ratio1) + EPS) + torch.log(1. / (1. + ratio2) + EPS)
                neg_js = 0.5 * (-output.mean(0) - torch.log(torch.tensor(4.)))
                neg_js = torch.clamp(neg_js, max=0)   # clip the residual positive part (if any)
                return neg_js.item()

            if self.divergence == 'kls':
                ratio1 = self._dr(m_pdf_m_data, nb_pdf_m_data)
                ratio2 = self._dr(nb_pdf_nb_data, m_pdf_nb_data)

                output = torch.log(ratio1 + EPS) + torch.log(ratio2 + EPS)
                neg_kls = -output.mean(0)
                neg_kls = torch.clamp(neg_kls, max=0)   # clip the residual positive part (if any)
                return neg_kls.item()

            raise ValueError("Unknown divergence")

    def update(self, m_buffer, nb_buffer, all_pdf_nets, nb_idx):
        self.train()

        m_states, m_actions = m_buffer.get_sample(nbatches=2)
        nb_states, nb_actions = nb_buffer.get_sample(nbatches=2)

        dset = CustomMapDataset(torch.cat([m_states, m_actions], dim=1), torch.cat([nb_states, nb_actions], dim=1))
        dset_generator = torch.utils.data.DataLoader(dset, batch_size=512, shuffle=True)

        loss_val = 0.
        n = 0.
        for data in dset_generator:
            m_sa, nb_sa = data     # get mini-batch

            # get PDF from the mixture distribution over the neighbors
            nb_pdf_m_data = [all_pdf_nets[i].infer(m_sa).detach() for i in all_pdf_nets.keys() if i != self.rank]
            nb_pdf_m_data = sum(nb_pdf_m_data) / len(nb_pdf_m_data)

            nb_pdf_nb_data = [all_pdf_nets[i].infer(nb_sa).detach() for i in all_pdf_nets.keys() if i != self.rank]
            nb_pdf_nb_data = sum(nb_pdf_nb_data) / len(nb_pdf_nb_data)

            normalized_sa = obs_batch_normalize(torch.cat([m_sa, nb_sa], dim=0), update_rms=True, rms_obj=self.input_rms)
            m_sa, nb_sa = torch.split(normalized_sa, [m_sa.size(0), nb_sa.size(0)], dim=0)

            m_pdf_m_data = self._modifier(self.tower(m_sa))
            m_pdf_nb_data = self._modifier(self.tower(nb_sa))

            ratio1 = self._dr(nb_pdf_m_data, m_pdf_m_data)
            ratio2 = self._dr(m_pdf_nb_data, nb_pdf_nb_data)

            grad_penalty = self.compute_grad_pen(m_sa, nb_sa, lambda_=0.1)
            loss = -torch.log(1. / (1. + ratio1) + EPS) - torch.log(1. / (1. + ratio2) + EPS) + grad_penalty
            loss = loss.mean(0)

            loss_val += loss.item()
            n += 1

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.)
            self.optimizer.step()

        neg_divergence = self.eval_neg_divergence(m_states, m_actions, nb_states, nb_actions, all_pdf_nets[nb_idx])
        return loss_val/n, neg_divergence

    def infer(self, sa):
        self.eval()
        with torch.no_grad():
            sa = obs_batch_normalize(sa, update_rms=False, rms_obj=self.input_rms)
            pdf = self._modifier(self.tower(sa))
        return pdf

    def predict_batch_rewards(self, idx, m_pdf, rollouts):
        assert idx != self.rank
        obs = rollouts.raw_obs[:-1].view(-1, self.ob_dim)
        acs = rollouts.actions.view(-1, self.ac_dim)
        sa = torch.cat([obs, acs], dim=1)
        pdf = self.infer(sa)

        if self.divergence == 'js':
            ratio = self._dr(m_pdf, pdf)
            rewards = -torch.log(1. / (1. + ratio) + EPS)

        elif self.divergence == 'kls':
            ratio = self._dr(pdf, m_pdf)
            rewards = -ratio - torch.log(ratio + EPS)

        else: raise ValueError("Unknown divergence")

        rewards = rewards.view(rollouts.num_steps, -1, 1)
        rollouts.aux_rewards[idx].copy_(rewards)

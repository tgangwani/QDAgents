import torch
import torch.nn as nn
import torch.nn.functional as F
from qd_agents.utils.common_utils import obs_batch_normalize, RunningMeanStd, init

class Discriminator(nn.Module):
    def __init__(self, ob_dim, ac_dim, hidden_dim, rank):
        super().__init__()

        self.rank = rank
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        input_dim = ob_dim + ac_dim
        actv = nn.Tanh
        gain = nn.init.calculate_gain('tanh')

        init_ = lambda m, bias=0: init(m, nn.init.xavier_normal_, lambda x: nn.init.
                constant_(x, bias), gain=gain)

        self.tower = nn.Sequential(
                init_(nn.Linear(input_dim, hidden_dim)), actv(),
                init_(nn.Linear(hidden_dim, hidden_dim)), actv(),
                init_(nn.Linear(hidden_dim, 1)))

        self.input_rms = RunningMeanStd(shape=input_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-3)

    def update(self, pq_buffer, rollouts, num_grad_steps):
        self.train()

        pqb_gen = pq_buffer.data_gen_finite(len(pq_buffer)//num_grad_steps)
        policy_data_gen = rollouts.feed_forward_generator(fetch_normalized=False,
                advantages=None, aux_advantages=None, mini_batch_size=rollouts.num_steps//num_grad_steps)

        loss_val = 0
        n = 0
        for _ in range(num_grad_steps):

            policy_batch = next(policy_data_gen)
            pqb_batch = next(pqb_gen)

            pqb_state, pqb_action, _ = pqb_batch
            policy_state, policy_action = policy_batch[0], policy_batch[2]

            policy_sa = torch.cat([policy_state, policy_action], dim=1)
            pqb_sa = torch.cat([pqb_state, pqb_action], dim=1)
            normalized_sa = obs_batch_normalize(torch.cat([policy_sa, pqb_sa], dim=0), update_rms=True, rms_obj=self.input_rms)
            policy_sa, pqb_sa = torch.split(normalized_sa, [policy_state.size(0), pqb_state.size(0)], dim=0)

            policy_logits = self.tower(policy_sa)
            pqb_logits = self.tower(pqb_sa)

            pqb_loss = F.binary_cross_entropy_with_logits(pqb_logits, torch.ones_like(pqb_logits))
            policy_loss = F.binary_cross_entropy_with_logits(policy_logits, torch.zeros_like(policy_logits))
            loss = pqb_loss + policy_loss

            loss_val += loss.item()
            n += 1

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.)
            self.optimizer.step()

        return loss_val / n

    def predict_batch_rewards(self, rollouts):
        obs = rollouts.raw_obs[:-1].view(-1, self.ob_dim)
        acs = rollouts.actions.view(-1, self.ac_dim)
        sa = torch.cat([obs, acs], dim=1)
        sa = obs_batch_normalize(sa, update_rms=False, rms_obj=self.input_rms)

        with torch.no_grad():
            self.eval()
            s = self.tower(sa).sigmoid()
            rewards = - (1 - s + 1e-6).log()
            rewards = rewards.view(rollouts.num_steps, -1, 1)

        rollouts.aux_rewards[self.rank].copy_(rewards)

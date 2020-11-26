import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class PPO():
    def __init__(self,
            rank, comm, comm_size,
            actor_critic,
            clip_param,
            ppo_epoch,
            num_mini_batch,
            value_loss_coef,
            si_wt, svpg_expl_wt,
            lr=None,
            max_grad_norm=None):

        self.rank = rank
        self.comm = comm
        self.comm_size = comm_size
        self.neighbors = [i for i in range(self.comm_size) if i != self.rank]

        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        # weight coefficients
        self.si_wt = si_wt     #  weight on the self-imitation term in the policy-gradient
        self.svpg_expl_wt = svpg_expl_wt           # weight on the exploration term in SVPG

        self.actor_params = [p for n, p in self.actor_critic.named_parameters() if not n.__contains__('critic')]
        self.critic_params = [p for n, p in self.actor_critic.named_parameters() if n.__contains__('critic')]
        self.actor_optimizer = optim.Adam(self.actor_params, lr=lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic_params, lr=lr, eps=1e-5)

        # buffer to store the gradient (policy parameters) from neighbors
        self.actor_pcnt = sum(p.numel() for p in self.actor_params if p.requires_grad)
        self.nb_grad_buffer = {i : np.zeros(self.actor_pcnt, dtype=np.float32) for i in self.neighbors}

    def update(self, rollouts, kernel_vals, anneal_coef):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        aux_advantages = [rollouts.aux_returns[i][:-1] - rollouts.aux_value_preds[i][:-1] for i in range(self.comm_size)]

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        for i in range(self.comm_size):
            aux_advantages[i] = (aux_advantages[i] - aux_advantages[i].mean()) / (aux_advantages[i].std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0

        for _ in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(fetch_normalized=True,
                    advantages=advantages, aux_advantages=aux_advantages, num_mini_batch=self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, arctanh_actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, \
                   aux_value_preds_batch_list, aux_return_batch_list, aux_adv_targ_list = sample

                # Reshape to do in a single forward pass for all steps
                values, aux_values, action_log_probs, _, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, arctanh_actions_batch)

                logratio = torch.clamp(action_log_probs - old_action_log_probs_batch, min=-10., max=10.)
                ratio = torch.exp(logratio)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                surr1 = ratio * aux_adv_targ_list[self.rank]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * aux_adv_targ_list[self.rank]
                action_loss = -torch.min(surr1, surr2).mean() * self.si_wt + action_loss * (1 - self.si_wt)
                action_loss_epoch += action_loss.item()

                self.actor_optimizer.zero_grad()
                action_loss.backward(retain_graph=True)
                grad_si = parameters_to_vector((p.grad for p in self.actor_params)).detach()  # self-imitation gradient
                assert grad_si.size(0) == self.actor_pcnt
                self.actor_optimizer.zero_grad()

                grad_exploration = torch.zeros(self.actor_pcnt)
                grad_exploitation = grad_si.clone()

                for count, i in enumerate(self.neighbors):
                    surr1 = ratio * aux_adv_targ_list[i]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * aux_adv_targ_list[i]
                    action_loss = -torch.min(surr1, surr2).mean()
                    self.actor_optimizer.zero_grad()
                    action_loss.backward(retain_graph=(count != len(self.neighbors)-1))
                    grad_jsd = parameters_to_vector((p.grad for p in self.actor_params)).detach()
                    self.actor_optimizer.zero_grad()
                    grad_exploration += kernel_vals[i] * grad_jsd
                    self.comm.Isend(grad_si.numpy(), dest=i, tag=77)

                for i in self.neighbors:
                    self.comm.Recv(self.nb_grad_buffer[i], source=i, tag=77)
                    grad_nb = torch.from_numpy(self.nb_grad_buffer[i])
                    assert grad_nb.size(0) == self.actor_pcnt
                    grad_exploitation += kernel_vals[i] * grad_nb

                self.comm.Barrier()

                # gradient averaging
                omega = anneal_coef if self.svpg_expl_wt is None else self.svpg_expl_wt
                grad_avg = grad_exploration * omega + grad_exploitation * (1 - omega)
                grad_avg /= (sum(kernel_vals.values()) + 1)

                # actor-update
                self.actor_optimizer.zero_grad()
                vector_to_parameters(grad_avg, (p.grad for p in self.actor_params))
                nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm)
                self.actor_optimizer.step()

                value_pred_clipped = value_preds_batch + \
                        torch.max(torch.min(values - value_preds_batch, self.clip_param*torch.abs(value_preds_batch)), -self.clip_param*torch.abs(value_preds_batch))
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                # add losses from auxiliary critics
                for i in range(self.comm_size):
                    aux_value_pred_clipped = aux_value_preds_batch_list[i] + \
                            torch.max(torch.min(aux_values[i] - aux_value_preds_batch_list[i], \
                            self.clip_param*torch.abs(aux_value_preds_batch_list[i])), -self.clip_param*torch.abs(aux_value_preds_batch_list[i]))
                    aux_value_losses = (aux_values[i] - aux_return_batch_list[i]).pow(2)
                    aux_value_losses_clipped = (aux_value_pred_clipped - aux_return_batch_list[i]).pow(2)
                    value_loss += 0.5 * torch.max(aux_value_losses, aux_value_losses_clipped).mean()

                value_loss = value_loss * self.value_loss_coef
                value_loss_epoch += value_loss.item()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_params, (1 + self.comm_size) * self.max_grad_norm)
                self.critic_optimizer.step()

        num_updates = self.ppo_epoch * self.num_mini_batch
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch

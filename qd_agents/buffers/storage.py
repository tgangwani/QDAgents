import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from qd_agents.utils.common_utils import obs_batch_normalize, RunningMeanStd

def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

class RolloutStorage:
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
            recurrent_hidden_state_size, gamma, gae_lambda, comm_size):
        self.raw_obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.normalized_obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)

        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.aux_rewards= torch.zeros(comm_size, num_steps, num_processes, 1)

        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.aux_value_preds = torch.zeros(comm_size, num_steps + 1, num_processes, 1)

        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.aux_returns = torch.zeros(comm_size, num_steps + 1, num_processes, 1)

        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        self.arctanh_actions = torch.zeros(num_steps, num_processes, action_shape)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.comm_size = comm_size

        self.ob_rms = RunningMeanStd(shape=obs_shape)
        self.num_steps = num_steps
        self.step = 0

    def insert(self, obs, recurrent_hidden_states, actions, arctanh_actions, action_log_probs,
               value_preds, aux_value_preds, rewards, masks, bad_masks):
        self.raw_obs[self.step + 1].copy_(obs)
        self.normalized_obs[self.step + 1].copy_(
                obs_batch_normalize(obs, update_rms=True, rms_obj=self.ob_rms))
        self.recurrent_hidden_states[self.step +
                                     1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.arctanh_actions[self.step].copy_(arctanh_actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)

        for i in range(self.comm_size):
            assert self.aux_value_preds[i, self.step].size() == aux_value_preds[i].size()
            self.aux_value_preds[i, self.step].copy_(aux_value_preds[i])

        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.raw_obs[0].copy_(self.raw_obs[-1])
        self.normalized_obs[0].copy_(self.normalized_obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_adv_tdlam(self, next_value, next_aux_value):
        self._compute_adv_tdlam_aux(next_aux_value)
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            gae = gae * self.bad_masks[step + 1]
            self.returns[step] = gae + self.value_preds[step]

    def _compute_adv_tdlam_aux(self, next_aux_value):
        for i in range(self.comm_size):
            self.aux_value_preds[i][-1] = next_aux_value[i]
            gae = 0
            for step in reversed(range(self.aux_rewards[i].size(0))):
                delta = self.aux_rewards[i][step] + self.gamma * self.aux_value_preds[i][step + 1] * self.masks[step + 1] - self.aux_value_preds[i][step]
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                gae = gae * self.bad_masks[step + 1]
                self.aux_returns[i][step] = gae + self.aux_value_preds[i][step]

    def feed_forward_generator(self,
                               fetch_normalized,
                               advantages,
                               aux_advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:

            if fetch_normalized:
                obs_batch = self.normalized_obs[:-1].view(-1, *self.normalized_obs.size()[2:])[indices]
            else:
                obs_batch = self.raw_obs[:-1].view(-1, *self.raw_obs.size()[2:])[indices]

            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            arctanh_actions_batch = self.arctanh_actions.view(-1, self.arctanh_actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            aux_value_preds_batch_list = [self.aux_value_preds[i][:-1].view(-1, 1)[indices] for i in range(self.comm_size)]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            aux_return_batch_list = [self.aux_returns[i][:-1].view(-1, 1)[indices] for i in range(self.comm_size)]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]

            if advantages is None:
                adv_targ = None
                aux_adv_targ_list = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]
                aux_adv_targ_list = [aux_advantages[i].view(-1, 1)[indices] for i in range(self.comm_size)]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, arctanh_actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, \
                aux_value_preds_batch_list, aux_return_batch_list, aux_adv_targ_list

import torch
from qd_agents.rl.ppo import PPO
from qd_agents.utils.envs import make_vec_envs
from qd_agents.networks.actor_critic import ActorCritic
from qd_agents.buffers.storage import RolloutStorage
from qd_agents.utils.common_utils import OrderedDefaultDict, obs_batch_normalize

class RLAgent():
    def __init__(self, args):

        self.envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                             args.log_dir, allow_early_resets=True)
        assert len(self.envs.observation_space.shape) == 1 and \
                len(self.envs.action_space.shape) == 1, \
                "Expected flat observation and action spaces. Consider adding a wrapper."
        assert self.envs.action_space.__class__.__name__ == "Box", "Continous action-space expected."

        self.obs_dim = self.envs.observation_space.shape[0]
        self.acs_dim = self.envs.action_space.shape[0]

        self.actor_critic_ntwk = ActorCritic(
            self.envs.observation_space.shape,
            self.envs.action_space,
            base_kwargs={'comm_size': args.comm_size})

        self.rl_algo = PPO(
            args.rank,
            args.comm,
            args.comm_size,
            self.actor_critic_ntwk,
            args.ppo_clip_param,
            args.ppo_epochs,
            args.num_mini_batch,
            args.value_loss_coef,
            args.si_wt,
            args.svpg_expl_wt,
            lr=args.policy_lr,
            max_grad_norm=args.max_grad_norm)

        self.num_steps = args.num_steps
        self.rollouts = RolloutStorage(self.num_steps, args.num_processes,
                                  self.envs.observation_space.shape, self.envs.action_space,
                                  self.actor_critic_ntwk.recurrent_hidden_state_size,
                                  args.gamma, args.gae_lambda, args.comm_size)

        self.latest_trajs = OrderedDefaultDict()
        self.num_finished_trajs = 0

        obs = self.envs.reset()
        self.rollouts.raw_obs[0].copy_(obs)
        self.rollouts.normalized_obs[0].copy_(obs)

    def sample_init_states(self, num):
        states = torch.zeros((num, self.obs_dim))
        for i in range(num):
            states[i] = self.envs.reset()
        return states

    def collect_rollout_batch(self, episode_returns):
        for step in range(self.num_steps):
            # Sample actions
            with torch.no_grad():
                value, aux_value, action, arctanh_action, action_log_prob, recurrent_hidden_states, _ = self.actor_critic_ntwk.act(
                    self.rollouts.normalized_obs[step], self.rollouts.recurrent_hidden_states[step],
                    self.rollouts.masks[step])

            obs, reward, dones, infos = self.envs.step(action)

            self.latest_trajs[self.num_finished_trajs]['states'].append(self.rollouts.raw_obs[step])
            self.latest_trajs[self.num_finished_trajs]['actions'].append(action)
            self.latest_trajs[self.num_finished_trajs]['arctanh_actions'].append(arctanh_action)
            self.latest_trajs[self.num_finished_trajs]['rewards'].append(reward)

            if dones.sum() > 0:
                for info in infos:
                    if 'episode' in info.keys():
                        episode_returns.append(info['episode']['r'])
                        self.num_finished_trajs += 1

            masks = torch.FloatTensor([[0.0] if done else [1.0] for done in dones])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

            self.rollouts.insert(obs, recurrent_hidden_states, action, arctanh_action,
                    action_log_prob, value, aux_value, reward, masks, bad_masks)

    def update(self, kernel_vals, anneal_coef):
        #  Compute GAE and TD-lambda estimates, and then update actor-critic parameters
        with torch.no_grad():
            next_value, next_aux_value = self.actor_critic_ntwk.get_value(
                self.rollouts.normalized_obs[-1], self.rollouts.recurrent_hidden_states[-1],
                self.rollouts.masks[-1])
        self.rollouts.compute_adv_tdlam(next_value, next_aux_value)

        value_loss, action_loss = self.rl_algo.update(self.rollouts, kernel_vals, anneal_coef)
        self.rollouts.after_update()
        return value_loss, action_loss

    def get_action(self, states, deterministic):
        with torch.no_grad():
            normalized_states = obs_batch_normalize(states, update_rms=False, rms_obj=self.rollouts.ob_rms)
            return self.actor_critic_ntwk.act(normalized_states, rnn_hxs=None, masks=None, deterministic=deterministic)[2].detach()

    def get_actor_params(self):
        return (p for n, p in self.actor_critic_ntwk.named_parameters() if not n.__contains__('critic'))

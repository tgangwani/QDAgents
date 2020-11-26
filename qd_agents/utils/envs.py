import os
import gym
import torch
from qd_agents.utils import baselines_support
from qd_agents.utils.vec_env import VecEnvWrapper

def make_env(env_id, seed, rank, log_dir, allow_early_resets):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)

        if hasattr(gym.envs, 'atari') and isinstance(
                env.unwrapped, gym.envs.atari.atari_env.AtariEnv):
            raise NotImplementedError("Code only tested for MuJoCo locomotion.")

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = baselines_support.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)

        return env

    return _thunk

def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  log_dir,
                  allow_early_resets):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        from qd_agents.utils.vec_env.shmem_vec_env import ShmemVecEnv
        envs = ShmemVecEnv(envs, context='fork')
    else:
        from qd_agents.utils.vec_env.dummy_vec_env import DummyVecEnv
        envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs)
    return envs

class TimeLimitMask(gym.Wrapper):
    """
    Checks whether done was caused my timit limits or not
    """
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class VecPyTorch(VecEnvWrapper):
    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float()
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float()
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

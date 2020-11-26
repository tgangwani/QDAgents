from collections import deque
import random
from copy import deepcopy
import numpy as np
import torch
from qd_agents.utils.common_utils import RunningMeanStd

class BatchFIFO:

    def __init__(self, capacity, obs_dim=None):
        self.capacity = capacity
        self.num_batches = 0
        self.buffer_ = deque()
        self.obs_rms = RunningMeanStd(shape=obs_dim) if obs_dim is not None else None

    def get_sample(self, nbatches):
        if self.num_batches < nbatches:
            return self.process(random.sample(self.buffer_, self.num_batches))

        # sampling w/o replacement. Always include the most recent batch!
        batches = [self.buffer_[-1]] + random.sample(self.buffer_, nbatches-1)
        return self.process(batches)

    @staticmethod
    def process(batches):
        assert len(batches) >= 1
        obs = np.copy(batches[0]['obs'])
        acs = np.copy(batches[0]['acs'])
        next_obs = np.copy(batches[0]['next_obs'])
        masks = np.copy(batches[0]['masks'])
        for batch in batches[1:]:
            obs = np.concatenate((obs, batch['obs']), axis=0)
            acs = np.concatenate((acs, batch['acs']), axis=0)
            next_obs = np.concatenate((next_obs, batch['next_obs']), axis=0)
            masks = np.concatenate((masks, batch['masks']), axis=0)
        return torch.from_numpy(obs), torch.from_numpy(acs), torch.from_numpy(next_obs), torch.from_numpy(masks)

    def size(self):
        return self.capacity

    def add(self, obs, acs, masks):

        if self.obs_rms is not None:
            self.obs_rms.update(obs)

        batch = dict(obs=deepcopy(obs[:-1]), acs=deepcopy(acs), next_obs=deepcopy(obs[1:]), masks=deepcopy(masks))
        if self.num_batches < self.capacity:
            self.buffer_.append(batch)
            self.num_batches += 1
        else:
            self.buffer_.popleft()
            self.buffer_.append(batch)

    def count(self):
        return self.num_batches

    def erase(self):
        self.buffer_ = deque()
        self.num_batches = 0

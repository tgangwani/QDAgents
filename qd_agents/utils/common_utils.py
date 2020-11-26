import os
from collections import defaultdict, OrderedDict
import numpy as np
import torch
import torch.utils.data

class CustomMapDataset(torch.utils.data.Dataset):
    def __init__(self, *tnsrs):
        self.tnsrs = tnsrs
        l = [x.shape[0] for x in self.tnsrs]
        assert len(set(l)) == 1, "All input tensors should have the same number of elements"

    def __len__(self):
        return self.tnsrs[0].shape[0]

    def __getitem__(self, index):
        return [x[index] for x in self.tnsrs]

class OrderedDefaultDict(OrderedDict):
    def __missing__(self, key):
        self[key] = defaultdict(list)
        return self[key]

def get_dr_limits():
    """
    Set the clipping range for the density ratio estimates
    """
    dr_min = 0.05
    dr_max = 10.
    return dr_min, dr_max

def obs_batch_normalize(obs_tnsr, update_rms, rms_obj):
    """
    Use this function for a batch of 1-D tensors only
    """
    obs_tnsr_np = obs_tnsr.numpy()
    if update_rms:
        rms_obj.update(obs_tnsr_np)

    obs_normalized_np = np.clip((obs_tnsr_np - rms_obj.mean) / np.sqrt(rms_obj.var + 1e-8), -10., 10.)
    obs_normalized_tnsr = torch.FloatTensor(obs_normalized_np).view(obs_tnsr.size())
    return obs_normalized_tnsr

class RunningMeanStd:
    """
    https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon * np.ones(1, dtype=np.float32)

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / (count + batch_count)
    new_var = M2 / (count + batch_count)
    new_count = batch_count + count

    return new_mean, new_var, new_count

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def get_new_dir(root):
    os.makedirs(root, exist_ok=True)
    numd = str(len(os.listdir(root)))
    dst_dir = os.path.join(root, 'run'+numd)
    assert not os.path.exists(dst_dir), 'Directory already exists {}'.format(dst_dir)
    os.makedirs(dst_dir)
    return dst_dir+'/'

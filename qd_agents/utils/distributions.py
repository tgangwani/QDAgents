import numpy as np
import torch
import torch.nn as nn

LOG_SIG_MIN = -10
LOG_SIG_MAX = 2

class TanhNormal:
    """
    x ~ tanh(normal(mu, std))
    """
    def __init__(self, normal_mean, normal_std):
        self._wrapped_normal = torch.distributions.Normal(normal_mean, normal_std)

    def log_probs(self, arctanh_actions):
        """
        We use a numerically stable formula for log(1 - tanh(x)^2), adapted from
        https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L73
        Formula: log(1 - tanh(x)^2) = 2 * (log(2) - x - softplus(-2x))
        """
        offset = 2 * (torch.from_numpy(np.log([2], dtype=np.float32)) - arctanh_actions - torch.nn.functional.softplus(-2. * arctanh_actions))
        lp = self._wrapped_normal.log_prob(arctanh_actions).sum(dim=1, keepdim=True) - offset.sum(dim=1, keepdim=True)
        return lp

    def mode(self):
        m = self._wrapped_normal.mean
        return m, torch.tanh(m)

    def rsample(self):
        """
        sample w/ reparameterization
        """
        z = self._wrapped_normal.rsample()
        return z, torch.tanh(z)

    def sample(self):
        """
        sample w/o reparameterization
        """
        z = self._wrapped_normal.sample().detach()
        return z, torch.tanh(z)

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.fc_mean = nn.Linear(num_inputs, num_outputs)
        self.logstd = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        action_logstd = self.logstd.expand_as(action_mean)
        action_logstd = torch.clamp(action_logstd, LOG_SIG_MIN, LOG_SIG_MAX)
        return TanhNormal(action_mean, action_logstd.exp())

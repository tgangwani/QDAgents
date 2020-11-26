import numpy as np
import torch.nn as nn
from qd_agents.utils.common_utils import init
from qd_agents.utils.distributions import DiagGaussian

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_space, base_kwargs=None, kwargs=None):
        super().__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if kwargs is None:
            kwargs = {}

        self.base = MLPBase(obs_shape[0], **base_kwargs)
        self.dist = DiagGaussian(self.base.output_size, action_space.shape[0], **kwargs)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError()

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, aux_value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        arctanh_action, action = dist.mode() if deterministic else dist.rsample()
        action_log_probs = None if deterministic else dist.log_probs(arctanh_action.detach())
        return value, aux_value, action, arctanh_action, action_log_probs, rnn_hxs, dist

    def get_value(self, inputs, rnn_hxs, masks):
        value, aux_value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value, aux_value

    def evaluate_actions(self, inputs, rnn_hxs, masks, _action, arctanh_action):
        value, aux_value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(arctanh_action)
        return value, aux_value, action_log_probs, rnn_hxs, dist

class NNBase(nn.Module):
    def __init__(self, recurrent, _recurrent_input_size, hidden_size, comm_size):
        super().__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self.comm_size = comm_size

        if recurrent:
            raise NotImplementedError("Current implementation does not support recurrent policy architectures.")

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        raise NotImplementedError("Current implementation does not support recurrent policy architectures.")

class MLPBase(NNBase):
    def __init__(self, num_inputs, comm_size, recurrent=False, hidden_size=64):
        super().__init__(recurrent, num_inputs, hidden_size, comm_size)

        if recurrent:
            raise NotImplementedError("Current implementation does not support recurrent policy architectures.")

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, 1)))

        # define auxiliary critic networks
        # for an MPI ensemble of size k, there are k auxiliary critics, one to
        # compute the state-value w/ the self-imitation rewards, and (k-1) to
        # compute the state-value w/ the SVPG exploration rewards
        self.aux_critics = nn.ModuleList([
                    nn.Sequential(
                        init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
                        init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                        init_(nn.Linear(hidden_size, 1)))
                    for _ in range(comm_size)
                ])

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        aux_critics_vals = [self.aux_critics[i](x) for i in range(self.comm_size)]
        return self.critic(x), aux_critics_vals, self.actor(x), rnn_hxs

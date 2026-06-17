from __future__ import annotations

import numpy as np
from typing import Any, NoReturn

import torch
import torch.nn as nn
from torch.distributions import Normal
from tensordict import TensorDict
from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.modules.him_estimator import HIMEstimator

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape, device):  # shape:the dimension of input data
        self.n = 1e-4
        self.uninitialized = True
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)

    def update(self, x):
        count = self.n
        batch_count = x.size(0)
        tot_count = count + batch_count

        old_mean = self.mean.clone()
        delta = torch.mean(x, dim=0) - old_mean

        self.mean = old_mean + delta * batch_count / tot_count
        m_a = self.var * count
        m_b = x.var(dim=0) * batch_count
        M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
        self.var = M2 / tot_count
        self.n = tot_count

class Normalization:
    def __init__(self, shape, device='cuda:0'):
        self.running_ms = RunningMeanStd(shape=shape, device=device)

    def __call__(self, x, update=False):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:  
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (torch.sqrt(self.running_ms.var) + 1e-4)

        return x

class HIMActorCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [512, 256, 128],
        critic_hidden_dims: tuple[int] | list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        obs_term_dims: list[int] | None = None,
        history_length: int | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(HIMActorCritic, self).__init__()

        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.state_dependent_std = state_dependent_std

        # HIM history layout
        # IsaacLab concatenates observation terms TERM-MAJOR and flattens each term's
        # history oldest->newest, i.e. the actor obs is laid out as
        #     [term0_t0..term0_t{H-1}, term1_t0..term1_t{H-1}, ...].
        # HIM needs a TIME-MAJOR layout where each timestep holds the full one-step obs
        # (this is the "sequence by time" processing used by HIMLoco / Go2Arm).
        # `obs_term_dims` is the per-single-step dimension of each policy obs term in
        # concatenation order; it is used by `_to_time_major` to reorder term-major ->
        # time-major. When it is None we assume the policy obs is a single term whose
        # history is already contiguous in time (term-major == time-major for one term).
        self.num_actions = num_actions
        self.obs_term_dims = list(obs_term_dims) if obs_term_dims is not None else None
        if self.obs_term_dims is not None:
            self.num_one_step_obs = sum(self.obs_term_dims)
            assert num_actor_obs % self.num_one_step_obs == 0, (
                f"Actor obs dim ({num_actor_obs}) is not divisible by the one-step obs dim "
                f"({self.num_one_step_obs}) implied by obs_term_dims={self.obs_term_dims}."
            )
            self.history_size = num_actor_obs // self.num_one_step_obs
            if history_length is not None:
                assert history_length == self.history_size, (
                    f"history_length ({history_length}) disagrees with the history size "
                    f"({self.history_size}) inferred from num_actor_obs / sum(obs_term_dims)."
                )
        else:
            self.history_size = history_length if history_length is not None else 6
            assert num_actor_obs % self.history_size == 0, (
                f"Actor obs dim ({num_actor_obs}) is not divisible by history_length "
                f"({self.history_size}). Provide `obs_term_dims` or a matching `history_length`."
            )
            self.num_one_step_obs = num_actor_obs // self.history_size
        num_one_step_obs = self.num_one_step_obs

        mlp_input_dim_a = num_one_step_obs + 3 + 16

        # Actor
        self.actor = MLP(mlp_input_dim_a, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        # Actor observation normalization
        # Note: the normalizer operates on the full (history) actor obs, so it must be
        # sized to `num_actor_obs`, not the actor MLP input.
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Estimator
        self.estimator = HIMEstimator(temporal_steps=self.history_size, num_one_step_obs=num_one_step_obs)

        print(f'Estimator: {self.estimator.encoder}')

        # Action noise
        # HIM (like HIMLoco) only supports a learned scalar std parameter.
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            raise ValueError("HIMActorCritic does not support state_dependent_std.")
        if self.noise_std_type != "scalar":
            raise ValueError(
                f"HIMActorCritic only supports noise_std_type='scalar', got '{self.noise_std_type}'."
            )
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def _to_time_major(self, obs_flat: torch.Tensor) -> torch.Tensor:
        """Reorder a term-major actor obs into a time-major history tensor.

        IsaacLab lays out the actor obs term-major (each term's history flattened
        oldest->newest):
            ``[term0_t0..term0_t{H-1}, term1_t0..term1_t{H-1}, ...]``
        HIM expects a time-major layout where each timestep holds the full one-step obs.

        Args:
            obs_flat: Flattened actor obs of shape ``(N, num_actor_obs)``.

        Returns:
            Tensor of shape ``(N, history_size, num_one_step_obs)`` ordered oldest->newest;
            the last time index (``[:, -1, :]``) is the most recent observation.
        """
        n = obs_flat.shape[0]
        if self.obs_term_dims is None:
            # Single term: its history is already contiguous in time.
            return obs_flat.reshape(n, self.history_size, self.num_one_step_obs)
        h = self.history_size
        chunks = []
        offset = 0
        for d in self.obs_term_dims:
            # (N, H*d) term-major block -> (N, H, d)
            chunks.append(obs_flat[:, offset : offset + h * d].reshape(n, h, d))
            offset += h * d
        # Concatenate term features per timestep -> (N, H, num_one_step_obs)
        return torch.cat(chunks, dim=-1)

    def get_actor_obs_history(self, obs: TensorDict) -> torch.Tensor:
        """Return the normalized, time-major, flattened actor obs history ``(N, H*one_step)``."""
        obs_flat = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self._to_time_major(obs_flat).flatten(1)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        # Reorder term-major obs into a time-major history (oldest->newest)
        hist = self._to_time_major(obs)  # (N, history_size, num_one_step_obs)
        obs_history = hist.flatten(1)  # (N, history_size * num_one_step_obs) time-major
        current_obs = hist[:, -1, :]  # (N, num_one_step_obs) most recent timestep
        with torch.no_grad():
            vel, latent = self.estimator(obs_history)
        actor_input = torch.cat((current_obs, vel, latent), dim=-1)
        # Compute mean
        mean = self.actor(actor_input)
        # Create distribution
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self._update_distribution(obs)
        return self.distribution.sample()
    
    def act_inference(self, obs_history: TensorDict) -> torch.Tensor:
        obs = self.actor_obs_normalizer(self.get_actor_obs(obs_history))
        hist = self._to_time_major(obs)  # (N, history_size, num_one_step_obs)
        vel, latent = self.estimator(hist.flatten(1))
        actor_input = torch.cat((hist[:, -1, :], vel, latent), dim=-1)
        return self.actor(actor_input)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)
    
    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True
    
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
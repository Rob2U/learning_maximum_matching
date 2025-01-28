from typing import Any, Callable, List, Optional, Tuple, Type

import numpy as np
import torch
import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3 import PPO
from torch import nn


class SingleLayerNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        layer_dim_pi: int = 64,
        layer_dim_vf: int = 64,
    ) -> None:
        super().__init__()

        # IMPORTANT:
        self.latent_dim_pi = layer_dim_pi
        self.latent_dim_vf = layer_dim_vf

        self.policy_net = nn.Sequential(nn.Linear(feature_dim, layer_dim_pi), nn.ReLU())
        self.value_net = nn.Sequential(nn.Linear(feature_dim, layer_dim_vf), nn.ReLU())

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass in all the networks.

        Args:
            features (torch.Tensor): The input features (output of the features extractor)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output of the actor and critic networks
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)  # type: ignore

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)  # type: ignore


class DoubleLayerNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        layer_dim_pi: int = 64,
        layer_dim_vf: int = 64,
    ) -> None:
        super().__init__()

        # IMPORTANT:
        self.latent_dim_pi = layer_dim_pi
        self.latent_dim_vf = layer_dim_vf

        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, layer_dim_pi),
            nn.ReLU(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, layer_dim_vf),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass in all the networks.

        Args:
            features (torch.Tensor): The input features (output of the features extractor)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output of the actor and critic networks
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)  # type: ignore

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)  # type: ignore


class MLPNetwork(nn.Module):
    def __init__(
        self,
        net_arch: List[int],
        feature_dim: int = 256,
        activation_fn: Type[nn.Module] = nn.ReLU,
        share_features_extractor: bool = True,
    ) -> None:
        super().__init__()

        # IMPORTANT:
        self.latent_dim_pi = net_arch[-1]
        self.latent_dim_vf = net_arch[-1]

        # Build the network
        self.policy_net = self.build_network(net_arch, feature_dim, activation_fn)
        if share_features_extractor:
            self.value_net = self.policy_net
        else:
            self.value_net = self.build_network(net_arch, feature_dim, activation_fn)

        self.share_features_extractor = share_features_extractor

    def build_network(
        self, net_arch: List[int], feature_dim: int, activation_fn: Type[nn.Module]
    ) -> nn.Sequential:
        input_layer_size = feature_dim
        layers = [nn.Linear(input_layer_size, net_arch[0]), activation_fn()]
        prev_layer_size = net_arch[0]
        for layer_size in net_arch[1:]:
            layers += [nn.Linear(prev_layer_size, layer_size), activation_fn()]
            prev_layer_size = layer_size

        return nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass in all the networks.

        Args:
            features (torch.Tensor): The input features (output of the features extractor)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output of the actor and critic networks
        """

        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)  # type: ignore

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)  # type: ignore


class TransformerNetwork(nn.Module):
    """We could also use a Transformer network instead of a MLP network."""

    pass


class CustomActorCriticPolicy(MaskableActorCriticPolicy):
    entropy: float

    def __init__(
        self,
        observation_space: spaces.Space[spaces.MultiDiscrete],
        action_space: spaces.Space[spaces.MultiDiscrete],
        lr_schedule: Callable[[float], float],
        feature_dim: int = 64,
        layer_dim_pi: int = 64,
        layer_dim_vf: int = 64,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False  # IDK what this is but this was set in the docs
        self.feature_dim = feature_dim
        self.layer_dim_pi = layer_dim_pi
        self.layer_dim_vf = layer_dim_vf

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MLPNetwork(
            net_arch=[self.layer_dim_pi] * 5,
            feature_dim=self.feature_dim,
            activation_fn=nn.ReLU,
            share_features_extractor=self.share_features_extractor,
        )  # type: ignore

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,  # type: ignore
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :param action_masks: Action masks to apply to the action distribution
        :return: action, value and log probability of the action
        """
        # COPIED FROM MaskableActorCriticPolicy.forward() and adjusted to log entropies

        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        entropy = distribution.entropy()
        if entropy is not None:
            self.entropy = entropy.item()

        return actions, values, log_prob

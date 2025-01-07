from typing import Any, Callable, Tuple

import torch
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


class TransformerNetwork(nn.Module):
    """We could also use a Transformer network instead of a MLP network."""

    pass


class CustomActorCriticPolicy(MaskableActorCriticPolicy):
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
        self.mlp_extractor = DoubleLayerNetwork(
            feature_dim=self.feature_dim,
            layer_dim_pi=self.layer_dim_pi,
            layer_dim_vf=self.layer_dim_vf,
        )  # type: ignore

import math

import gymnasium as gym
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class PositionalEncoding(nn.Module):
    """See: https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)  # type: ignore


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Space[gym.spaces.MultiDiscrete],
        features_dim: int = 32,
        d_model: int = 32,
        nhead: int = 4,
        num_blocks: int = 4,
        num_instructions: int = 12,
        program_length: int = 32,
        device: str = "cpu",
    ):
        """Transformer-based features extractor for the MSTCode environment.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            features_dim (int, optional): The dimension of the features extracted by the transformer. Defaults to 32.
            d_model (int, optional): The number of expected features in the input. Defaults to 32.
            nhead (int, optional): The number of heads in the multiheadattention models. Defaults to 4.
            num_blocks (int, optional): The number of blocks in the encoder. Defaults to 4.
            num_instructions (int, optional): The number of instructions in the environment. Defaults to 12.
            program_length (int, optional): The maximum length of the program. Defaults to 32.
        """
        super().__init__(observation_space, features_dim)
        self.num_instructions = num_instructions
        self.program_length = program_length

        self.positional_encoding = PositionalEncoding(
            d_model=d_model, dropout=0.0, max_len=program_length + 1
        )
        self.embedding = nn.Linear(
            num_instructions, d_model, device=device
        )  # (num_instructions) -> (d_model): Linear layer to embed the one-hot encoded instructions
        self.class_token = nn.Parameter(
            torch.randn(1, 1, d_model, device=device)
        )  # (1, 1, d_model): Learnable class token

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, device=device
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_blocks
        )  # Transformer encoder (num_layers many blocks)
        self.fc_out = nn.Linear(
            d_model, features_dim, device=device
        )  # (d_model) -> (features_dim): Linear layer to output the features

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch_size, program_length * num_instructions) -> one hot encoded and concatenated
        x = observations.float().view(-1, self.program_length, self.num_instructions)
        x = self.embedding(
            x
        )  # (batch_size, program_length, num_instructions) -> (batch_size, program_length, d_model)

        x = torch.cat(
            (self.class_token.repeat(x.size(0), 1, 1), x), dim=1
        )  # (batch_size, program_length, d_model) -> (batch_size, program_length + 1, d_model)

        # Positional encoding should be added here
        x = x.transpose(
            0, 1
        )  # (batch_size, program_length + 1, d_model) -> (program_length + 1, batch_size, d_model)
        x = self.positional_encoding(
            x
        )  # (program_length + 1, batch_size, d_model) -> (program_length + 1, batch_size, d_model)
        x = x.transpose(
            0, 1
        )  # (program_length + 1, batch_size, d_model) -> (batch_size, program_length + 1, d_model)

        x = self.transformer_encoder(
            x
        )  # (batch_size, program_length + 1, d_model) -> (batch_size, program_length + 1, d_model)
        x = x[
            :, 0
        ]  # select the class token (batch_size, program_length + 1, d_model) -> (batch_size, d_model)
        x = self.fc_out(x)
        return x


if __name__ == "__main__":
    from ..environment.environment import MSTCodeEnvironment

    observation_space = gym.spaces.MultiDiscrete([12] * 32)
    feature_dim = 64
    d_model = 64
    nhead = 4
    num_blocks = 4
    num_instructions = 12
    program_length = 32

    policy_kwargs = dict(
        features_extractor_class=TransformerFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_blocks=num_blocks,
            num_instructions=num_instructions,
            program_length=program_length,
        ),
        net_arch=[
            feature_dim,
            feature_dim,
        ],  # MLP layers for actor and critic after the transformer
    )

    gym.register("MSTCode-v0", entry_point=MSTCodeEnvironment)
    env = gym.make("MSTCode-v0")
    model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=1000)
    model.save("ppo_transformer_mstcode")

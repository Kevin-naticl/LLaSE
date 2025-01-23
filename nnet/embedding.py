# Copyright    2023                             (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim_model = dim_model

        self.dropout = torch.nn.Dropout(p=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim_model, dim_model),  
            nn.ReLU(),
            nn.Linear(dim_model, dim_model), 
        )
        self.init_weights()

    def init_weights(self, gain: float = 1.0):
        torch.nn.init.normal_(self.word_embeddings.weight, std=0.02 * gain)

    @property
    def weight(self) -> torch.Tensor:
        return self.word_embeddings.weight

    def forward(self, x: torch.Tensor):
        X = self.mlp(x)
        X = self.dropout(X)

        return X

class TokenEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        vocab_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim_model = dim_model

        self.dropout = torch.nn.Dropout(p=dropout)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.dim_model)
        self.init_weights()

    def init_weights(self, gain: float = 1.0):
        torch.nn.init.normal_(self.word_embeddings.weight, std=0.02 * gain)

    @property
    def weight(self) -> torch.Tensor:
        return self.word_embeddings.weight

    def forward(self, x: torch.Tensor):
        X = self.word_embeddings(x)
        X = self.dropout(X)

        return X


class SinePositionalEmbedding(nn.Module):
    def __init__(self, dim_model: int):
        super().__init__()
        self.dim_model = dim_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        pos = (
            torch.arange(0, seq_len, device=x.device, dtype=torch.float32)
            .unsqueeze(1)
            .repeat(1, self.dim_model)
        )
        dim = (
            torch.arange(
                0, self.dim_model, device=x.device, dtype=torch.float32
            )
            .unsqueeze(0)
            .repeat(seq_len, 1)
        )
        # div = torch.exp(-math.log(10000) * (2 * (dim // 2) / self.dim_model))
        div = torch.exp(-math.log(10000) * (2 * (torch.div(dim, 2)) / self.dim_model))

        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])

        output = x.unsqueeze(-1) if x.ndim == 2 else x

        return output + pos.unsqueeze(0)


class SinePositionalEmbedding_V2(nn.Module):
    def __init__(self, feature_dim: int, max_seq_len: int = 1024, temperature=10000):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.positional_embeddings = self._generate_positional_embeddings()

    def _generate_positional_embeddings(self):
        div_term = torch.exp(
            torch.arange(0, self.feature_dim, 2).float()
            * -(torch.log(torch.tensor(self.temperature)) / self.feature_dim)
        )

        positions = torch.arange(0, self.max_seq_len).float().unsqueeze(1)
        pos_emb = torch.zeros(self.max_seq_len, self.feature_dim)
        pos_emb[:, 0::2] = torch.sin(positions * div_term)
        pos_emb[:, 1::2] = torch.cos(positions * div_term)

        return pos_emb.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError("Input sequence length exceeds maximum sequence length.")

        pos_emb = self.positional_embeddings[:, :seq_len, :]
        pos_emb = pos_emb.to(x.device)

        output = x + pos_emb

        return output


if __name__=="__main__":
    x = torch.rand([4, 199, 1024])
    pos_emb = SinePositionalEmbedding_V2(1024)
    out = pos_emb(x)
    print(out.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys,os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embedding import SinePositionalEmbedding, TokenEmbedding
from transformers import LlamaConfig, LlamaForCausalLM

NUM_AUDIO_TOKENS = 65536 #Number of Xcodec codebook
    
class AdaptiveLayerNorm(nn.Module):
    r"""Adaptive Layer Normalization"""

    def __init__(self, d_model, norm) -> None:
        super(AdaptiveLayerNorm, self).__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def forward(self, input: torch.Tensor, embedding: torch.Tensor = None) -> torch.Tensor:
        if isinstance(input, tuple):
            input, embedding = input        
            weight, bias = torch.split(
                self.project_layer(embedding),
                split_size_or_sections=self.d_model,
                dim=-1,
            )
            return (weight * self.norm(input) + bias, embedding)

        weight, bias = torch.split(
            self.project_layer(embedding),
            split_size_or_sections=self.d_model,
            dim=-1,
        )

        return weight * self.norm(input) + bias

class LLM_AR(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int
    ):
        super().__init__()

        self.audio_linear = nn.Linear(1024, d_model)
        self.audio_position = SinePositionalEmbedding(d_model)
        self.stage_embedding = TokenEmbedding(d_model, 1)
        self.adaLN = AdaptiveLayerNorm(d_model, norm=nn.LayerNorm(d_model))

        self.Llama_config = LlamaConfig(
            hidden_size=d_model,
            intermediate_size=d_model * 4,
            num_attention_heads=nhead,
            num_hidden_layers=num_layers,
            dropout_rate=0.1,
            attention_dropout=0.1,
            is_decoder=True,
            use_cache=True
        )

        self.lm = LlamaForCausalLM(config=self.Llama_config)
        self.predict_layer = nn.Linear(d_model, NUM_AUDIO_TOKENS)

    def forward(
        self,
        y: torch.Tensor,
    ) -> torch.Tensor:

        y_emb = self.audio_linear(y)  # [B, T, D]
        y_pos = self.audio_position(y_emb)  # [B, T, D]

        stage_embedding = self.stage_embedding(torch.tensor(0, device=y_pos.device))
        y_pos = self.adaLN(y_pos, stage_embedding)

        outputs = self.lm(inputs_embeds=y_pos, output_hidden_states=True)
        y_dec = outputs.hidden_states[-1]  # [B, T, D]

        logits = self.predict_layer(y_dec)  # [B, T, NUM_AUDIO_TOKENS]

        logits = logits.transpose(-1, -2)  # [B, NUM_AUDIO_TOKENS, T]

        return logits

if __name__=="__main__":
    # for test
    model = LLM_AR(d_model=1024, nhead=8, num_layers=12, task="SE")
    ce_loss = nn.CrossEntropyLoss()
    x = torch.randn([2,199,1024])
    label = torch.from_numpy(np.random.randint(0, 300, size=[2,1,199]))
    logits = model(x)
    print(logits.shape)
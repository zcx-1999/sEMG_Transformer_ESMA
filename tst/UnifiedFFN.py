from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.utils import clamp_probs


def relaxed_bernoulli_logits(probs, temperature):
    probs = clamp_probs(probs)
    uniforms = clamp_probs(torch.rand(probs.shape, dtype=probs.dtype, device=probs.device))
    return (uniforms.log() - (-uniforms).log1p() + probs.log() - (-probs).log1p()) / temperature


def bernoulli_sample(probs, temperature=1.0):
    logits = relaxed_bernoulli_logits(probs, temperature)
    y_soft = torch.sigmoid(logits)
    y_hard = (logits > 0.0).float()
    ret = y_hard.detach() - y_soft.detach() + y_soft
    return ret
class UnifiedFFN(nn.Module):
    """Position-wise Feed Forward Network block from Attention is All You Need.

    Apply two linear transformations to each input, separately but indetically. We
    implement them as 1D convolutions. Input and output have a shape (batch_size, d_model).

    Parameters
    ----------
    d_model:
        Dimension of input tensor.
    d_ff:
        Dimension of hidden layer, default is 2048.
    """

    def __init__(self,
                 d_model: int,
                 d_ff: Optional[int] = 1024,act_layer=nn.GELU, drop=0., theta=0., ffn_indicators=None):
        """Initialize the PFF block."""
        super().__init__()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        if ffn_indicators is None:
            self._linear1 = nn.Linear(d_model, d_ff)
            self._linear2 = nn.Linear(d_ff, d_model)
            # Threshold parameters
            self.register_parameter('ffn_thresholds', nn.Parameter(torch.tensor([theta] * d_ff)))

            # The indicators
            self.register_buffer('assigned_indicator_index', nn.Parameter(torch.zeros(d_ff)))
            self.fine_tuning = False
        else:
            self._linear1 = nn.Linear(d_model, ffn_indicators.nonzero().shape[0])
            self._linear2 = nn.Linear(ffn_indicators.nonzero().shape[0], d_model)

            self.fine_tuning = True
    def forward(self, x):
        if not self.fine_tuning:
            return self.search_forward(x)
        else:
            return self.finetune_forward(x)

    def search_forward(self, x):

        ffn_probs = F.sigmoid(self.ffn_thresholds)
        if self.training:
            ffn_indicators = bernoulli_sample(ffn_probs)
        else:
            ffn_indicators = (ffn_probs > 0.5).float()
        #print(ffn_probs)
        x = self._linear1(x)
        x = self.act(x)
        #print(ffn_probs)
        x = ffn_indicators.unsqueeze(0).unsqueeze(0) * x
        #print(x)
        x = self.drop(x)
        x = self._linear2(x)
        x = self.drop(x)

        # We derive the FFN indicators by expectation, and
        # ffn_indicators are kept to calculate complexity loss
        self.ffn_indicators = (ffn_probs > 0.5).float() - torch.sigmoid(
            ffn_probs - 0.5).detach() + torch.sigmoid(ffn_probs - 0.5)
        # print(self.ffn_indicators)
        # print(self.ffn_indicators.sum())

        self.register_buffer('assigned_indicator_index', self.ffn_indicators)
        return x, ffn_indicators, ffn_probs

    def finetune_forward(self, x):
        x = self._linear1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self._linear2(x)
        x = self.drop(x)

        return x, None, None


    #
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """Propagate forward the input through the PFF block.
    #
    #     Apply the first linear transformation, then a relu actvation,
    #     and the second linear transformation.
    #
    #     Parameters
    #     ----------
    #     x:
    #         Input tensor with shape (batch_size, K, d_model).
    #
    #     Returns
    #     -------
    #         Output tensor with shape (batch_size, K, d_model).
    #     """
    #     return self._linear2(F.relu(self._linear1(x)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tst.multiHeadAttention import MultiHeadAttention, MultiHeadAttentionChunk,EMSA,MultiHeadAttentionWindow
from tst.UnifiedFFN import UnifiedFFN
from tst.positionwiseFeedForward import PositionwiseFeedForward


class Encoder(nn.Module):
    """Encoder block from Attention is All You Need.

    Apply Multi Head Attention block followed by a Point-wise Feed Forward block.
    Residual sum and normalization are applied at each step.

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Swict between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = None,
                 Unified=False,
                 ffn_indicators=None):  ## chunk_mode: str = 'chunk'
        """Initialize the Encoder block"""
        super().__init__()
        self.Unified = Unified
        self.ffn_indicators = ffn_indicators
        chunk_mode_modules = {
            'chunk': MultiHeadAttentionChunk,
            'EMSA': EMSA,
            'window': MultiHeadAttentionWindow
        }
        if chunk_mode in chunk_mode_modules.keys():
            MHA = chunk_mode_modules[chunk_mode]
        elif chunk_mode is None:
            MHA = MultiHeadAttention
        else:
            raise NameError(
                f'chunk_mode "{chunk_mode}" not understood. Must be one of {", ".join(chunk_mode_modules.keys())} or None.')
        self._selfAttention = MHA(d_model, q, v, h, attention_size)
        self._unifiedFFN = UnifiedFFN(d_model, ffn_indicators=self.ffn_indicators)
        self._feedForward = PositionwiseFeedForward(d_model)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)

        self._dopout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate the input through the Encoder block.

        Apply the Multi Head Attention block, add residual and normalize.
        Apply the Point-wise Feed Forward block, add residual and normalize.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_model).
        """
        # Self attention
        #x = self._layerNorm1(x)
        residual = x
        x = self._selfAttention(query=x, key=x, value=x)

        x = self._dopout(x)
        x = self._layerNorm1(x + residual)

        # Feed forward
        residual = x
        ffn_indicators = None
        if self.Unified == False:
            x = self._feedForward(x)
            #print("sssss")
        else:
            x, ffn_indicators, _ = self._unifiedFFN(x)
        x = self._dopout(x)
        x = self._layerNorm2(x + residual)

        return x,ffn_indicators

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        return self._selfAttention.attention_map

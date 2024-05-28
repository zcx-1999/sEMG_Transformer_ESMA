from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn import init
from tst.utils import generate_local_map_mask
# #

def exists(val):
    return val is not None
beta = 1/4
def get_dist_mask_tile(sentence_len):
    """
        Calculate Cauchy matrix
        :param sentence_len: Length of attention matrix
        :return: dis_mask: Returns a matrix in which the elements obey the Cauchy distribution
        """
    row, col = torch.meshgrid(torch.arange(sentence_len), torch.arange(sentence_len))
    dis_mask = (row - col).abs()
    #print(dis_mask)
    dis_mask =  1 /(1 + beta*np.power(dis_mask, 2))
    #print(dis_mask.shape)
    return dis_mask
# Laplace
# dis_alpha=1/5
# def get_dist_mask_tile(sentence_len):
#     row, col = torch.meshgrid(torch.arange(sentence_len), torch.arange(sentence_len))
#     # row,col = row *4,col*4
#     dis_mask = (row - col).abs()
#     #print(dis_mask.shape)
#     dis_mask = np.exp(-dis_alpha*dis_mask.float())
#     #print(dis_mask)
#     return -dis_mask

class MultiHeadAttention(nn.Module):
    """Multi Head Attention block from Attention is All You Need.

    Given 3 inputs of shape (batch_size, K, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, K, d_model).

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
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None):
        """Initialize the Multi Head Block."""
        super().__init__()

        self._h = h
        self._attention_size = attention_size

        # Query, keys and value matrices
        self._W_q = nn.Linear(d_model, q*self._h)
        self._W_k = nn.Linear(d_model, q*self._h)
        self._W_v = nn.Linear(d_model, v*self._h)

        # Output linear function
        self._W_o = nn.Linear(self._h*v, d_model)

        # Score placeholder
        self._scores = None

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:
        """Propagate forward the input through the MHB.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated 
        and returned with shape (batch_size, K, d_model).

        Parameters
        ----------
        query:
            Input tensor with shape (batch_size, K, d_model) used to compute queries.
        key:
            Input tensor with shape (batch_size, K, d_model) used to compute keys.
        value:
            Input tensor with shape (batch_size, K, d_model) used to compute values.
        mask:
            Mask to apply on scores before computing attention.
            One of ``'subsequent'``, None. Default is None.

        Returns
        -------
            Self attention tensor with shape (batch_size, K, d_model).
        """
        K = query.shape[1]


        queries = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        keys = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        values = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(K)
        # Compute local map mask
        # if self._attention_size is not None:
        #     #print(self._attention_size)
        #     #attention_mask = generate_local_map_mask(K, self._attention_size, mask_future=False, device=self._scores.device)
        #     self._scores = self._scores.masked_fill(self._attention_size, float('-inf'))
        # # Compute future mask
        # if mask == "subsequent":
        #     self._scores = self._scores.masked_fill(self._scores, float('-inf'))

        # Apply sotfmax


        self._scores = F.softmax(self._scores, dim=-1)


        attention = torch.bmm(self._scores, values)

        # Concatenat the heads
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)


        return self_attention

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        if self._scores is None:
            raise RuntimeError(
                "Evaluate the model once to generate attention map")
        return self._scores


class MultiHeadAttentionChunk(MultiHeadAttention):
    """Multi Head Attention block with chunk.

    Given 3 inputs of shape (batch_size, K, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, K, d_model).
    Queries, keys and values are divided in chunks of constant size.

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
    chunk_size:
        Size of chunks to apply attention on. Last one may be smaller (see :class:`torch.Tensor.chunk`).
        Default is 168.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 chunk_size: Optional[int] = 50,
                 **kwargs):
        """Initialize the Multi Head Block."""
        super().__init__(d_model, q, v, h, attention_size, **kwargs)

        self._chunk_size = chunk_size
        # Score mask for decoder
        self._future_mask = nn.Parameter(torch.triu(torch.ones((self._chunk_size, self._chunk_size)), diagonal=1).bool(),
                                         requires_grad=False)
        self.scale = 1 / (h ** 0.5)
        # allowing for attending to nothing (null function)
        if self._attention_size is not None:
            self._attention_mask = nn.Parameter(generate_local_map_mask(self._chunk_size, self._attention_size),
                                                requires_grad=False)

    def squash(self,input_tensor):
        '''Squashes an input Tensor so it has a magnitude between 0-1.
           param input_tensor: a stack of capsule inputs, s_j
           return: a stack of normalized, capsule output vectors, v_j
           '''
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)  # normalization coeff
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)
        return output_tensor
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:
        """Propagate forward the input through the MHB.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated 
        and returned with shape (batch_size, K, d_model).

        Parameters
        ----------
        query:
            Input tensor with shape (batch_size, K, d_model) used to compute queries.
        key:
            Input tensor with shape (batch_size, K, d_model) used to compute keys.
        value:
            Input tensor with shape (batch_size, K, d_model) used to compute values.
        mask:
            Mask to apply on scores before computing attention.
            One of ``'subsequent'``, None. Default is None.

        Returns
        -------
            Self attention tensor with shape (batch_size, K, d_model).
        """
        K = query.shape[1]
        n_chunk = K // self._chunk_size

        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0).chunk(n_chunk, dim=1), dim=0)
        keys = torch.cat(torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0).chunk(n_chunk, dim=1), dim=0)
        values = torch.cat(torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0).chunk(n_chunk, dim=1), dim=0)

        # Scaled Dot Product
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(self._chunk_size)
        self._scores.mul_(self.scale)
        ##

        ##
        ## dir_masked
        ##
        # dir_mask = torch.triu(torch.ones((self._scores.shape[1], self._scores.shape[1])), diagonal=1).bool().cuda()
        # self._scores = self._scores.masked_fill(dir_mask, float('-inf'))
        # print(self._scores)


        ## 最后一点，如何证明关注局部特征有效
        ##
        ## Compute local map mask
        ##
        # if self._attention_size is not None:
        #     self._scores = self._scores.masked_fill(self._attention_mask, float('-inf'))


        ##
        ##Caual masked
        ##
        i,j = self._scores.shape[-2:]
        causal_mask = torch.ones(i, j, device=self._scores.device, dtype=torch.bool).triu(j - i + 1)
        self._scores = self._scores.masked_fill(causal_mask,float('-inf'))
        #
        #
        # # # Compute future mask
        # if mask == "subsequent":
        #     # future_mask = torch.triu(torch.ones((self._chunk_size, self._chunk_size)), diagonal=1).bool()
        #     # future_mask = future_mask.to(self._scores.device)
        #     self._scores = self._scores.masked_fill(self._future_mask, float('-inf'))

        # Apply softmax

        self._scores = F.softmax(self._scores, dim=-1)
        #print(self._scores)
        attention = torch.bmm(self._scores, values)
        # attention = self.squash(attention)
        #print(attention.shape)
        # Concatenat the heads
        attention_heads = torch.cat(torch.cat(attention.chunk(
            n_chunk, dim=0), dim=1).chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention

class EMSA(MultiHeadAttention):

    def __init__(self, d_model:int, q:int, v:int, h:int, attention_size:int = None ,dropout:int=.1, H:int=10, W:int=20, ratio:int=2, apply_transform:bool=False,**kwargs):
        super().__init__(d_model, q, v, h, attention_size, **kwargs)
        self.H = H
        self.W = W
        # self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        # self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.ratio = ratio
        if (self.ratio > 1):
            self.sr = nn.Sequential()
            self.sr_conv = nn.Conv2d(d_model, d_model, kernel_size=ratio + 1, stride=ratio, padding=ratio // 2 )#,groups=d_model).cuda()
            self.sr_ln = nn.LayerNorm(d_model)

        self.apply_transform = apply_transform and h > 1
        if (self.apply_transform):
            self.transform = nn.Sequential()
            self.transform.add_module('conv', nn.Conv2d(h, h, kernel_size=1, stride=1))
            self.transform.add_module('softmax', nn.Softmax(-1))
            self.transform.add_module('in', nn.InstanceNorm2d(h))
        # self.up = nn.Sequential(
        #     #nn.Conv2d(d_model, ratio * ratio * d_model, kernel_size=3, stride=1, padding=1, groups=d_model),
        #     #nn.Conv2d(ratio * ratio * d_model, d_model, kernel_size=3, stride=1, padding=1, groups=d_model),
        #     nn.PixelShuffle(upscale_factor=ratio)
        # )
        # self.up_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.down = nn.Linear(d_model*2,d_model)
        #self.down = nn.Linear(200*8,200)
        #self.proj = nn.Linear(d_model, d_model)
        #self.d_model = d_model
        self.d_k = q
        self.d_v = v
        self._h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, query, key, value, mask=None, attention_weights=None):


        b_s, nq, c = query.shape
        nk = key.shape[1]

        q = self._W_q(query).view(b_s, -1, self._h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)

        if (self.ratio > 1):
            x = query.permute(0, 2, 1).view(b_s, c, self.H, self.W)  # bs,c,H,W
            # print(x.shape)
            x = self.sr_conv(x)  # bs,c,h,w
            # x = x.contiguous().view(b_s, c, -1).permute(0, 2, 1)  # bs,n',c
            x = x.view(b_s, c, -1).permute(0, 2, 1)  # bs,n',c
            x = self.sr_ln(x)
            k = self._W_k(x).view(b_s, -1, self._h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, n')
            v = self._W_v(x).view(b_s, -1, self._h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, n', d_v)
            # print(v.shape)
        else:
            k = self._W_k(key).view(b_s, nk, self._h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
            v = self._W_v(value).view(b_s, nk, self._h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        if (self.apply_transform):
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, n')
            att = self.transform(att)  # (b_s, h, nq, n')
        else:
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, n')
            # att = torch.matmul(q,k)
            att = torch.softmax(att, -1)  # (b_s, h, nq, n')

        if attention_weights is not None:
            att = att * attention_weights
        # if mask is not None:
        #     att = att.masked_fill(mask, -np.inf)

        # att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, -1, self._h * self.d_v)  # (b_s, nq, h*d_v)
        out = self._W_o(out)  # (b_s, nq, d_model)
        ## 使用conv_2d的卷积上采样
        # print(v.shape)
        # identity = v.transpose(-1, -2).reshape(b_s, c, self.W, self.W)
        # identity = self.down(self.up(identity).flatten(2)).transpose(1, 2)
        # out = self.proj(out + self.up_norm(identity))

        #identity = v.transpose(-1, -2)#.reshape(b_s, c, self.H // self.ratio, self.W // self.ratio)
        #identity = self.up(identity).transpose(-2, -3).flatten(2).transpose(-1, -2)
        #out = self.proj(out + self.up_norm(self.down(identity)))
        return out
class MultiHeadAttentionWindow(MultiHeadAttention):
    """Multi Head Attention block with moving window.

    Given 3 inputs of shape (batch_size, K, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, K, d_model).
    Queries, keys and values are divided in chunks using a moving window.

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
    window_size:
        Size of the window used to extract chunks.
        Default is 168
    padding:
        Padding around each window. Padding will be applied to input sequence.
        Default is 168 // 4 = 42.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 window_size: Optional[int] = 200,
                 padding: Optional[int] = 200 // 4,
                 **kwargs):
        """Initialize the Multi Head Block."""
        super().__init__(d_model, q, v, h, attention_size, **kwargs)

        self._window_size = window_size
        self._padding = padding
        self._q = q
        self._v = v

        # Step size for the moving window
        self._step = self._window_size - 2 * self._padding

        # Score mask for decoder
        self._future_mask = nn.Parameter(torch.triu(torch.ones((self._window_size, self._window_size)), diagonal=1).bool(),
                                         requires_grad=False)

        if self._attention_size is not None:
            self._attention_mask = nn.Parameter(generate_local_map_mask(self._window_size, self._attention_size),
                                                requires_grad=False)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:
        """Propagate forward the input through the MHB.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated
        and returned with shape (batch_size, K, d_model).

        Parameters
        ----------
        query:
            Input tensor with shape (batch_size, K, d_model) used to compute queries.
        key:
            Input tensor with shape (batch_size, K, d_model) used to compute keys.
        value:
            Input tensor with shape (batch_size, K, d_model) used to compute values.
        mask:
            Mask to apply on scores before computing attention.
            One of ``'subsequent'``, None. Default is None.

        Returns
        -------
            Self attention tensor with shape (batch_size, K, d_model).
        """
        batch_size = query.shape[0]

        # Apply padding to input sequence
        query = F.pad(query.transpose(1, 2), (self._padding, self._padding), 'replicate').transpose(1, 2)
        key = F.pad(key.transpose(1, 2), (self._padding, self._padding), 'replicate').transpose(1, 2)
        value = F.pad(value.transpose(1, 2), (self._padding, self._padding), 'replicate').transpose(1, 2)

        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        keys = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        values = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)

        # Divide Q, K and V using a moving window
        queries = queries.unfold(dimension=1, size=self._window_size, step=self._step).reshape((-1, self._q, self._window_size)).transpose(1, 2)
        keys = keys.unfold(dimension=1, size=self._window_size, step=self._step).reshape((-1, self._q, self._window_size)).transpose(1, 2)
        values = values.unfold(dimension=1, size=self._window_size, step=self._step).reshape((-1, self._v, self._window_size)).transpose(1, 2)

        # Scaled Dot Product
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(self._window_size)

        dist_mask_tile = get_dist_mask_tile(self._scores.shape[1])
        diagonal_zero_mask = np.ones([self._scores.shape[1], self._scores.shape[1]]) -  np.diag([1.0] * self._scores.shape[1])
        diagonal_zero_mask=torch.tensor(diagonal_zero_mask)
        dist_mask_tile=dist_mask_tile*diagonal_zero_mask
        self._scores *= dist_mask_tile.cuda()
        dir_mask = torch.triu(torch.ones((self._scores.shape[1], self._scores.shape[1])), diagonal=1).bool().cuda()
        # Compute local map mask
        if self._attention_size is not None:
            self._scores = self._scores.masked_fill(dir_mask, float('-inf'))

        # Compute future mask
        if mask == "subsequent":
            self._scores = self._scores.masked_fill(self._future_mask, float('-inf'))

        # Apply softmax
        self._scores = F.softmax(self._scores, dim=-1)

        attention = torch.bmm(self._scores, values)

        # Fold chunks back
        attention = attention.reshape((batch_size*self._h, -1, self._window_size, self._v))
        attention = attention[:, :, self._padding:-self._padding, :]
        attention = attention.reshape((batch_size*self._h, -1, self._v))

        # Concatenat the heads
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention

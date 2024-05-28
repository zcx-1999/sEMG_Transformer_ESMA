import torch
import torch.nn as nn
import torch.nn.functional as F

from tst.encoder import Encoder
from tst.decoder import Decoder
from tst.utils import generate_original_PE, generate_regular_PE
from base.loss_transfer import TransferLoss

class Transformer(nn.Module):
    """Transformer model from Attention is All You Need.

    A classic transformer model adapted for sequential data.
    Embedding has been replaced with a fully connected layer,
    the last layer softmax is now a sigmoid.

    Attributes
    ----------
    layers_encoding: :py:class:`list` of :class:`Encoder.Encoder`
        stack of Encoder layers.
    layers_decoding: :py:class:`list` of :class:`Decoder.Decoder`
        stack of Decoder layers.

    Parameters
    ----------
    d_input:
        Model input dimension.
    d_model:
        Dimension of the input vector.
    d_output:
        Model output dimension.
    q:
        Dimension of queries and keys.
    v:
        Dimension of values.
    h:
        Number of heads.
    N:
        Number of encoder and decoder layers to stack.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Switch between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    pe:
        Type of positional encoding to add.
        Must be one of ``'original'``, ``'regular'`` or ``None``. Default is ``None``.
    pe_period:
        If using the ``'regular'` pe, then we can define the period. Default is ``24``.
    """

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = None,  ##chunk
                 pe: str = None,
                 pe_period: int = 24,
                 Unified = False,
                 ffn_indicators=None):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()
        self._d_model = d_model
        self.Unified = Unified
        self.ffn_indicators = ffn_indicators
        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode,
                                                      Unified = self.Unified,
                                                      ffn_indicators=self.ffn_indicators) for _ in range(N)])
        self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode,
                                                      Unified = self.Unified,
                                                      ffn_indicators = self.ffn_indicators) for _ in range(N)])

        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

        self.name = 'transformer'
        # self.conv1d = nn.Conv1d(in_channels=10,out_channels=20,kernel_size=3,stride=1,padding=1)
        # self.max_pool_1d = nn.MaxPool1d(kernel_size=3,padding=1,stride=1)
        self.Linear1 = nn.Linear(200*10,10)

        self.tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate input through transformer

        Forward input through an embedding module,
        the encoder then decoder stacks, and an output module.

        Parameters
        ----------
        x:
            :class:`torch.Tensor` of shape (batch_size, K, d_input).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_output).
        """
        K = x.shape[1]

        # Embeddin module
        #print(x.shape)
        encoding = self._embedding(x)
        # print(encoding.shape)

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params).cuda()
            positional_encoding = positional_encoding.to(encoding.device).cuda()
            encoding.add_(positional_encoding).cuda()

        # Encoding stack
        list_encoding = []
        for layer in self.layers_encoding:
            encoding,ffn_indicators = layer(encoding)
            list_encoding.append(encoding)

        # Decoding stack
        decoding = encoding
        #
        # Add position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self._d_model)
            positional_encoding = positional_encoding.to(decoding.device)
            decoding.add_(positional_encoding).cuda()
        #
        for layer in self.layers_decoding:
            decoding = layer(decoding, encoding)

        # Output module
        output = self._linear(decoding)  ## decoding
        #output = self.tanh(output)
        #output = output.squeeze(-1)
        # output = torch.sigmoid(output)
        # print(fea.shape)
        # output = output.permute(0, 2, 1).type(torch.FloatTensor).cuda()
        # output = self.conv1d(output)
        # output = self.max_pool_1d(output)
        # #print(output.shape)
        # output = output.permute(0, 2, 1).type(torch.FloatTensor).cuda()
        output = self.Linear1(output.reshape(output.shape[0],output.shape[1]*output.shape[2]))
        #output = output[:,-1,:]

        return output,ffn_indicators

    def adapt_encoding(self, list_encoding, loss_type, train_type):
        ## train_type "last" : last hidden, "all": all hidden
        loss_all = torch.zeros(1).cuda()
        for i in range(len(list_encoding)):
            data = list_encoding[i]
            data_s = data[0:len(data)//2]
            data_t = data[len(data)//2:]
            criterion_transder = TransferLoss(
                loss_type=loss_type, input_dim=data_s.shape[2])
            if train_type == 'last':
                loss_all = loss_all + criterion_transder.compute(
                        data_s[:, -1, :], data_t[:, -1, :])
            elif train_type == "all":
                for j in range(data_s.shape[1]):
                    loss_all = loss_all + criterion_transder.compute(data_s[:, j, :], data_t[:, j, :])
            else:
                print("adapt loss error!")
        return loss_all


    def adapt_encoding_weight(self, list_encoding, loss_type, train_type, weight_mat=None):
        loss_all = torch.zeros(1).cuda()
        len_seq = list_encoding[0].shape[1]
        num_layers = len(list_encoding)
        if weight_mat is None:
            weight = (1.0 / len_seq *
                      torch.ones(num_layers, len_seq)).cuda()
        else:
            weight = weight_mat
        dist_mat = torch.zeros(num_layers, len_seq).cuda()
        for i in range(len(list_encoding)):
            data = list_encoding[i]
            data_s = data[0:len(data)//2]
            data_t = data[len(data)//2:]
            criterion_transder = TransferLoss(
                loss_type=loss_type, input_dim=data_s.shape[2])
            if train_type == 'last':
                loss_all = loss_all + criterion_transder.compute(
                        data_s[:, -1, :], data_t[:, -1, :])
            elif train_type == "all":
                for j in range(data_s.shape[1]):
                    loss_transfer = criterion_transder.compute(data_s[:, j, :], data_t[:, j, :])
                    loss_all = loss_all + weight[i, j] * loss_transfer
                    dist_mat[i, j] = loss_transfer 
            else:
                print("adapt loss error!")
        return loss_all, dist_mat, weight

    def update_weight_Boosting(self, weight_mat, dist_old, dist_new):
        epsilon = 1e-5
        dist_old = dist_old.detach()
        dist_new = dist_new.detach()
        ind = dist_new > dist_old + epsilon
        weight_mat[ind] = weight_mat[ind] * \
            (1 + torch.sigmoid(dist_new[ind] - dist_old[ind]))
        weight_norm = torch.norm(weight_mat, dim=1, p=1)
        weight_mat = weight_mat / weight_norm.t().unsqueeze(1).repeat(1, len(weight_mat[0]))
        return weight_mat



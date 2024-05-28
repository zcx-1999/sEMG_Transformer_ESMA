import torch
import torch.nn as nn
import numpy as np
from tst import Transformer
from tst.encoder import Encoder
from tst.utils import generate_original_PE, generate_regular_PE


class Muti_Trans(nn.Module):
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
                 chunk_mode: str = 'chunk',
                 pe: str = None,
                 pe_period: int = 200
                 ):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()
        self.streams = d_input
        self._d_model = d_model

        self.layers = nn.ModuleList()

        for i in range(self.streams):
            self.layers.append(
                Transformer(d_input=1, d_model=d_model, d_output=d_output, q=q, v=v, h=h, N=N, attention_size=attention_size,
                                  chunk_mode=chunk_mode,
                                  pe=pe, pe_period=0).cuda()
            )
        self.Linear = nn.Linear(120,10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_streams = []
        #print(x.shape)  ## [32,1,10,20]
        for idx in range(self.streams):
            p = x[:,:,idx]
            p = torch.unsqueeze(p,axis=2)
            #print(p.shape)
            lay_out,_ = self.layers[idx](p)
            #print(lay_out.shape)
            out_streams.append(lay_out)
            #print(idx)
        output = torch.cat(out_streams, dim=1)
        output = self.Linear(output)
        #print(output.shape)
        return output
if __name__ == "__main__":
    x = torch.randn(128,200,12).cuda()
    d_model = 16  # 32  Lattent dim
    q = 8  # Query size
    v = 8  # Value size
    h = 4  # 4   Number of heads
    N = 1  # Number of encoder and decoder to stack
    attention_size = 8 # Attention window size #8
    pe = "regular"  # Positional encoding   "original"  "regular"
    chunk_mode = 'window' ## "chunk"  "window"
    d_input = 12  # From dataset
    d_output = 10  # From dataset
    model = Muti_Trans(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size, chunk_mode=chunk_mode,
                        pe=pe, pe_period=0).cuda()
    fc_out = model(x)
    #fc_out,loss_tranferss,out_list,_ = model.forward_Boosting(x,None)
    print(fc_out)
    #print(fc_out.cpu().shape)



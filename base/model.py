import torch.nn.functional as F
from torch import nn

from base.tcn import TemporalConvNet


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        #self.linear1 = nn.Linear(num_channels[-1], output_size)
        self.linear1 = nn.Linear(2000, 256)
        self.linear2 = nn.Linear(256, output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (batch, Channel, Length)
        #print(y1.shape)
        o1 = self.linear1(y1.view([y1.shape[0],y1.shape[1]*y1.shape[2]])) #input should have dimension (batch,Length ,Channel)
        #print(o1.shape)
        o1 = self.linear2(o1)
        return o1

class TCN_huake(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN_huake, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear1 = nn.Linear(num_channels[-1], 16)
        self.linear2 = nn.Linear(16,output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (batch, Channel, Length)
        o1 = self.linear1(y1[:,:,-1])
        o2 = self.linear2(o1) #o2 (batchsize,inputsize,channel,)
        return o2


import torch
import torch.nn as nn
from torch.autograd import Variable

# from base.loss.LabelSmoothing import LabelSmoothing


class RNN(nn.Module):
    def __init__(self,input_size,output_size,rnn_model = 'GRU',hidden_size = 64,num_layers = 1,dropout = 0.0):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.out_putsize = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_model = rnn_model
        if rnn_model == 'GRU':
            self.RNN = nn.GRU(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,dropout=dropout,bidirectional=False,batch_first=True)
        elif rnn_model == 'LSTM':
            self.RNN = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,dropout=dropout,batch_first=True)
        elif rnn_model == 'BiLSTM':
            self.RNN = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,bidirectional=True,
                               batch_first=True)
        elif rnn_model == 'RNN':
            self.RNN = nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,dropout=dropout,batch_first=True)
        else:
            raise LookupError(' only support LSTM and GRU')
        self.fc  = nn.Linear(hidden_size*200,output_size)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.fc2 = nn.Linear(hidden_size*2,output_size)

    def forward(self,x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).float()).cuda()
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).float()).cuda()
        if self.rnn_model == "LSTM":
            x,_ = self.RNN(x,(h0,c0))
        else:
            x,_ = self.RNN(x,h0)
        #print(x.shape)
        # x = x[ :, -1 ,:]
        #print(x.shape)
        x = self.fc(x.reshape(x.shape[0],x.shape[1]*x.shape[2]))
        # if self.rnn_model == 'BiLSTM':
           # x = self.fc2(x)
        # else:
           # x = self.fc1(x)
        return x
if __name__ == "__main__":
    x = torch.randn(64,200,12).cuda()
    y = torch.randn(64,10).cuda()
    # loss = LabelSmoothing(smoothing=0.3)
    model = RNN(input_size=12,output_size=10,rnn_model='LSTM',hidden_size=128,num_layers=2,dropout=0.0).cuda()
    x = model(x.cuda())
    # loss_1 = loss(x,y)
    print(x)

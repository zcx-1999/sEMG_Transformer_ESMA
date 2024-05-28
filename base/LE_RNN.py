import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self,input_size,output_size,rnn_model = 'GRU',hidden_size = 64,num_layers = 1,dropout = 0.0):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.out_putsize = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        if rnn_model == 'LE-GRU':
            self.RNN = nn.GRU(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,dropout=dropout,bidirectional=False,batch_first=True)
        elif rnn_model == 'LE-LSTM':
            self.RNN = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,dropout=dropout,batch_first=True)
        elif rnn_model == 'LE-RNN':
            self.RNN = nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,dropout=dropout,batch_first=True)
        else:
            raise LookupError(' only support LSTM and GRU')
        self.fc = nn.Linear(hidden_size*200, output_size)
        # self.fc = nn.Linear(hidden_size,output_size)

    def forward(self,x,_ht):
        x,_ht = self.RNN(x,_ht)
        #print(x.shape)
        # x = x[ :, -1 ,:]
        #print(x.shape)
        x = self.fc(x.reshape(x.shape[0],x.shape[1]*x.shape[2]))
        # x = self.fc(x)
        return x,_ht
if __name__ == "__main__":
    x = torch.randn(64, 200, 12).cuda()
    model = RNN(input_size=12, output_size=10, rnn_model='LSTM', hidden_size=128, num_layers=2, dropout=0.0).cuda()
    x = model(x.cuda())
    print(x.shape)

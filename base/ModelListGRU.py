import torch
import torch.nn as nn
class ModelListGRU(nn.Module):
    def __init__(self,input_size,output_size,rnn_model = 'GRU',hidden_size = 64,num_layers = 1,dropout = 0.0):
        super(ModelListGRU, self).__init__()
        self.input_size = input_size
        self.out_putsize = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        modellist = nn.ModuleList()
        if rnn_model == 'GRU':
            self.RNN = nn.GRU(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,dropout=dropout,bidirectional=False,batch_first=True)
            modellist.append(self.RNN)
        elif rnn_model == 'LSTM':
            self.RNN = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,dropout=dropout)
        elif rnn_model == 'RNN':
            self.RNN = nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,dropout=dropout)

        else:
            raise LookupError(' only support LSTM and GRU')
        self.fc = nn.Linear(hidden_size, output_size)
        self.features = nn.Sequential(*modellist)
    #     self.init_layers()
    # def init_layers(self):
    #     for i in range(len(self.hiddens)):
    #         self.gate[i].weight.data.normal_(0, 0.05)  ##权重可以尝试改改 默认0.05
    #         self.gate[i].bias.data.fill_(0.0)
    def forward(self,x,his):
        x,_ht = self.features[0](x,his)
        #print(x.shape)
        x = x[ :, -1 ,:]
        #print(x.shape)
        # x = self.fc(x.reshape(x.shape[0],x.shape[1]*x.shape[2]))
        #x = self.fc(x)
        return x,_ht
if __name__ == "__main__":
    x = torch.randn(64,200,12).cuda()
    model = ModelListGRU(input_size=12,output_size=10,rnn_model='GRU',hidden_size=128,num_layers=2,dropout=0.0).cuda()
    x,_ = model(x.cuda(), None)
    print(x.shape)
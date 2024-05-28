import torch 
import torch.nn.functional as F
from torch import nn


class simple_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(simple_net, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc3 = nn.Linear(2000,10)
    def forward(self, inputs):
        y1 = self.fc1(inputs)
        y1 = self.fc2(y1)  
        o1 = self.fc3(y1.view([y1.shape[0],y1.shape[1]*y1.shape[2]])) 
        #print(o1.shape)
        # o1 = self.fc3(o1)
        return o1
if __name__ == "__main__":
    x = torch.randn(64, 200, 12).cuda()
    model = simple_net(input_size = 12,hidden_size = 128,output_size = 10).cuda()
    x = model(x.cuda())
    print(x.size())


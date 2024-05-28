import torch
import torch.nn as nn

class LabelSmoothing(nn.Module):
    """
    MSE loss with label smoothing
    """
    def __init__(self,smoothing = 0.1):
        super(LabelSmoothing, self).__init__()
        assert  0.0 < smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1 - smoothing
        self.MSEloss = nn.MSELoss()
    def forward(self,x,target):
        y_hat = torch.softmax(x,dim=2)
        #onehot = torch.zeros_like(x).scatter(1,target.view(-1,1),1)
        smooth_loss = -torch.log(y_hat).mean(dim=1)
        mseloss = self.MSEloss(x,target)
        loss = self.confidence*mseloss + smooth_loss
        return loss.mean()

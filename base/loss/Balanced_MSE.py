import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.distributions import MultivariateNormal as MVN


class BMCLossMD(_Loss):
    """
    Multi-Dimension version BMC, compatible with 1-D BMC
    """

    def __init__(self, init_noise_sigma):
        super(BMCLossMD, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss_md(pred, target, noise_var)
        return loss


def bmc_loss_md(pred, target, noise_var):
    I = torch.eye(pred.shape[-1]).cuda()
    logits = MVN(pred.unsqueeze(1), noise_var*I).log_prob(target.unsqueeze(0))
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())
    loss = loss * (2 * noise_var).detach()
    return loss
if __name__ == "__main__":
    x = torch.randn(64,128).cuda()
    y = torch.randn(64,128).cuda()
    loss = BMCLossMD(init_noise_sigma=0.1).cuda()  # Just like any torch.nn.xyzLoss()
    loss_1 = loss(x,y)
    print(loss_1)
    # Aggregate and call backward()
    #loss.mean().backward()
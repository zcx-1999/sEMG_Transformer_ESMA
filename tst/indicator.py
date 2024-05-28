import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from torch.distributions.utils import clamp_probs


def relaxed_bernoulli_logits(probs, temperature):
    probs = clamp_probs(probs)
    uniforms = clamp_probs(torch.rand(probs.shape, dtype=probs.dtype, device=probs.device))
    return (uniforms.log() - (-uniforms).log1p() + probs.log() - (-probs).log1p()) / temperature


def bernoulli_sample(probs, temperature=1.0):
    logits = relaxed_bernoulli_logits(probs, temperature)
    y_soft = torch.sigmoid(logits)
    y_hard = (logits > 0.0).float()
    ret = y_hard.detach() - y_soft.detach() + y_soft
    return ret
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class UnifiedMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., theta=0., ffn_indicators=None):
        super().__init__()

        self.in_features = in_features
        out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

        if ffn_indicators is None:

            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

            # Threshold parameters
            self.register_parameter('ffn_thresholds', nn.Parameter(torch.tensor([theta] * hidden_features)))

            # The indicators
            self.register_buffer('assigned_indicator_index', nn.Parameter(torch.zeros(hidden_features)))
            self.fine_tuning = False
        else:
            self.fc1 = nn.Linear(in_features, ffn_indicators.nonzero().shape[0])
            self.fc2 = nn.Linear(ffn_indicators.nonzero().shape[0], out_features)

            self.fine_tuning = True

    def forward(self, x):
        if not self.fine_tuning:
            return self.search_forward(x)
        else:
            return self.finetune_forward(x)

    def search_forward(self, x):

        ffn_probs = F.sigmoid(self.ffn_thresholds)
        if self.training:
            ffn_indicators = bernoulli_sample(ffn_probs)
        else:
            ffn_indicators = (ffn_probs > 0.5).float()

        x = self.fc1(x)
        x = self.act(x)
        print(ffn_probs)
        x = ffn_indicators.unsqueeze(0).unsqueeze(0) * x
        print(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        # We derive the FFN indicators by expectation, and
        # ffn_indicators are kept to calculate complexity loss
        self.ffn_indicators = (ffn_probs > 0.5).float() - torch.sigmoid(
            ffn_probs - 0.5).detach() + torch.sigmoid(ffn_probs - 0.5)

        self.register_buffer('assigned_indicator_index', self.ffn_indicators)
        return x, ffn_indicators, ffn_probs

    def finetune_forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x, None, None
if __name__ == '__main__':
    input = torch.randn(128, 200, 12)
    model = Mlp(in_features=12,hidden_features=1024,out_features=10)
    y = model(input)
    flops, params = profile(model, inputs=(input,))
    print(flops,params)
    print(y.shape)
    model = UnifiedMlp(in_features=12,hidden_features=1024,out_features=10)
    y, ffn_indicators, ffn_probs = model(input)
    #print(ffn_indicators)
    model = UnifiedMlp(in_features=12, hidden_features=1024, out_features=10,ffn_indicators=ffn_indicators)
    flops, params = profile(model, inputs=(input,))
    print(flops,params)
    # y,ffn_indicators, ffn_probs = model(input)

    print(y.shape)


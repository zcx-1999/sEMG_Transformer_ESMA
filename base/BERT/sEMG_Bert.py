import torch.nn as nn
import torch


# import torchsnooper
# @torchsnooper.snoop()
from base.BERT.Bert import BERT


class sEMG_BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=24, feature_dim=6, n_layers=12, attn_heads=12, dropout=0.1, use_se=False):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.name = "BERT"
        self.bert = BERT(vocab_size, hidden, feature_dim, n_layers, attn_heads, dropout, use_se=use_se)
        self.fc = nn.Linear(vocab_size * hidden * 1, 1 * 10)
        self.vocab_size = vocab_size

    def forward(self, x):
        x1 = self.bert(x)
        distr = x1
        x1_flatten = x1.view(x1.size(0), -1)
        output = self.fc(x1_flatten)
        output = output.view(x1.size(0), 1, 10)
        return output, distr


if __name__ == "__main__":
    sim_batch = torch.rand([8, 200, 12]).cuda()
    model = sEMG_BERT(vocab_size=200, hidden=128,feature_dim=1,n_layers=4,attn_heads=8).cuda()
    print(sim_batch.shape)
    output = model(sim_batch)
    print(output[1].shape)

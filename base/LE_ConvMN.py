import torch.nn as nn
import torch
# from utils.ConvLSTM import ConvLSTM
from base.LE_Conv import ConvLSTM


class LE_convMN(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_size, input_dim, output_dim, hidden_dim, kernel_size, num_layers, outpoint,
                 batch_first=False, bias=True, return_all_layers=False, withCBAM=False):
        super(LE_convMN, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        self.name = "LE_ConvMN"
        self.ConvMN = ConvLSTM(input_size, input_dim, output_dim, hidden_dim, kernel_size, num_layers, outpoint,
                               batch_first, bias, return_all_layers,)
        # self.fc1 = nn.Linear(1*10,2048)
        # self.fc2 = nn.Linear(2048,1*10)
        self.fc = nn.Linear(hidden_dim[-1], 1 * 10)

    def forward(self, input_tensor, hidden_state=None):
        # print(input_tensor.size())
        b, h, w, c = input_tensor.size()
        input_tensor = input_tensor.reshape([b, 1, c, h, w])
        output, hidden_state = self.ConvMN(input_tensor, hidden_state)
        # print(temp_tensor.shape)
        # temp_tensor = output.view(b, -1)
        # output = self.fc(temp_tensor)
        # print(output.shape)
        # output = self.fc1(output)
        # output = self.fc2(output)
        output = output.reshape(b, 1, 10)
        # print(output.shape)
        return output, tuple(hidden_state)
if __name__ == "__main__":
    x = torch.randn(64,200,12).cuda()
    model = LE_convMN((12, 200), 1, 1, [64, 32, 10], (3, 3), 3, 20, batch_first=True, bias=True,
                       return_all_layers=False, withCBAM=False).cuda()
    x,_ = model(x.cuda(), None)
    print(x.shape)

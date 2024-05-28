import torch
import torch.nn as nn
from torch.autograd import Variable
from tsmoothie import LowessSmoother,KalmanSmoother

from base.loss_transfer import TransferLoss
import torch.nn.functional as F
import numpy as np
class AdaRNN(nn.Module):
    """
    model_type:  'Boosting', 'AdaRNN'
    """

    def __init__(self, use_bottleneck=False, bottleneck_width=256, n_input=128, n_hiddens=[64, 64], n_output=6, dropout=0.0, len_seq=9, model_type='AdaRNN', trans_loss='mmd'):
        super(AdaRNN, self).__init__()
        self.use_bottleneck = use_bottleneck
        self.n_input = n_input
        self.num_layers = len(n_hiddens)
        self.hiddens = n_hiddens
        self.n_output = n_output
        self.model_type = model_type
        self.trans_loss = trans_loss
        self.len_seq = len_seq
        in_size = self.n_input

        features = nn.ModuleList()
        for hidden in n_hiddens:
            rnn = nn.GRU(
                input_size=in_size,
                num_layers=1,
                hidden_size=hidden,
                batch_first=True,
                ## 是否需要加偏执
                bias = True,
                bidirectional=False,
                dropout=dropout
            )
            #print(rnn.shape)
            #rnn = nn.LSTM(input_size=in_size,num_layers=1,hidden_size=hidden,batch_first=True,dropout=dropout,bias=True)
            # rnn =
            features.append(rnn)
            in_size = hidden
        self.features = nn.Sequential(*features)
        # self.attention_size = 10
        # #（4，30）
        # self.w_omega = Variable(torch.zeros(self.hidden_size * self.num_layers, self.attention_size)).cuda()
        # #（30）
        # self.u_omega = Variable(torch.zeros(self.attention_size)).cuda()

        if use_bottleneck == True:  # finance
            self.bottleneck = nn.Sequential(
                nn.Linear(n_hiddens[-1], bottleneck_width),
                #nn.Linear(bottleneck_width, bottleneck_width),
                ### 去掉正则化，CC系数效果会非常好
                #nn.BatchNorm1d(bottleneck_width),
                #nn.ReLU(),
                #nn.Dropout(),
            )
            ## 权重需要重新调整一下
            self.bottleneck[0].weight.data.normal_(0, 0.005) ### 默认0.005
            self.bottleneck[0].bias.data.fill_(0.1)   ### 默认0.1
            #self.bottleneck[1].weight.data.normal_(0, 0.005)  ## 默认0.005
            #self.bottleneck[1].bias.data.fill_(0.1)   ### 默认0.1
            self.fc = nn.Linear(bottleneck_width, n_output)
            torch.nn.init.xavier_normal_(self.fc.weight)
        else:
            self.fc_out = nn.Linear(n_hiddens[-1], self.n_output)
        #self.FC = nn.Linear(128*12,10)
        if self.model_type == 'AdaRNN':
            gate = nn.ModuleList()
            for i in range(len(n_hiddens)):
                gate_weight = nn.Linear(
                     len_seq * self.hiddens[i]*2, len_seq)  ### self.hiddens[i]*2, n_hiddens[-1])
                gate.append(gate_weight)
            self.gate = gate

            bnlst = nn.ModuleList()

            for i in range(len(n_hiddens)):
                bnlst.append(nn.BatchNorm1d(len_seq))   ## BN层可以尝试消除试试  BatchNorm1d()n_hiddens[-1]
            self.bn_lst = bnlst
            self.softmax = torch.nn.Softmax(dim=0)
            #self.init_layers()
            self.Linear = nn.Linear(n_hiddens[-1]*self.num_layers,self.n_output)

    def init_layers(self):
        for i in range(len(self.hiddens)):
            self.gate[i].weight.data.normal_(0, 0.05)  ##权重可以尝试改改 默认0.05
            self.gate[i].bias.data.fill_(0.0)

    def forward_pre_train(self, x, len_win=0):
        out = self.gru_features(x)
        #print(out)
        fea = out[0]
        if self.use_bottleneck == True:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            fc_out = self.fc_out(fea[:, -1, :]).squeeze()

        out_list_all, out_weight_list = out[1], out[2]
        out_list_s, out_list_t = self.get_features(out_list_all)
        loss_transfer = torch.zeros((1,)).cuda()
        ### O(N3)   len_win
        for i in range(len(out_list_s)):
            criterion_transder = TransferLoss(
                loss_type=self.trans_loss, input_dim=out_list_s[i].shape[2])
            h_start = 0
            ## len_win可以尝试修改一下
            #print(self.len_seq)
            #print("这是第i个list",i)
            ## 以下所有的len_seq全部改为  self.hiddens[-1]
            for j in range(h_start, self.hiddens[-1], 1):
                i_start = j - len_win if j - len_win >= 0 else 0
                i_end = j + len_win if j + len_win < self.len_seq else self.len_seq - 1
                ## 0 200
                #print("这是第j个数据",j)
                #print("i_start = ",i_start,"     i_end = ",i_end)
                for k in range(i_start, i_end + 1):
                    weight = out_weight_list[i][j] if self.model_type == 'AdaRNN' else 1 / (
                        self.hiddens[-1] - h_start) * (2 * len_win + 1)
                    loss_transfer = loss_transfer + weight * criterion_transder.compute(
                        out_list_s[i][:, j, :], out_list_t[i][:, k, :])
        return fc_out, loss_transfer, out_weight_list

    def gru_features(self, x, predict=False):
        x_input = x
        out = None
        out_lis = []
        out_weight_list = [] if (
             self.model_type == 'AdaRNN') else None
        hidden_state = None
        for i in range(self.num_layers):
            out, _ = self.features[i](x_input.float(),hidden_state)
            #hidden_state = _
            #print(i,x_input.shape)
            x_input = out ## out
            #x_input = torch.unsqueeze(x_input, dim=2)
            #print(i,x_input.shape)
            out_lis.append(out)
            if self.model_type == 'AdaRNN' and predict == False:
                out_gate = self.process_gate_weight(x_input, i)
                out_weight_list.append(out_gate)
        return out, out_lis, out_weight_list

    def process_gate_weight(self, out, index):
        x_s = out[0: int(out.shape[0]//2)]
        x_t = out[out.shape[0]//2: out.shape[0]]
        x_all = torch.cat((x_s, x_t), 2) ## ((x_s, x_t), 2)
        x_all = x_all.view(x_all.shape[0], -1)
        weight = torch.sigmoid(self.bn_lst[index](
            self.gate[index](x_all.float())))
        weight = torch.mean(weight, dim=0)
        #print(weight)
        res = self.softmax(weight).squeeze()   ## self.softmax(weight).squeeze()
        #print(res)
        return res

    def get_features(self, output_list):
        fea_list_src, fea_list_tar = [], []
        for fea in output_list:
            fea_list_src.append(fea[0: fea.size(0) // 2])
            fea_list_tar.append(fea[fea.size(0) // 2:])
        return fea_list_src, fea_list_tar

    # For Boosting-based
    def forward_Boosting(self, x, weight_mat=None):
        out = self.gru_features(x)
        fea = out[0]
        if self.use_bottleneck:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            fc_out = self.fc(fea_bottleneck).squeeze()
        else:
            #print(fea[:,-1,:].shape)
            fc_out = self.fc_out(fea[:, -1, :]).squeeze()

        out_list_all = out[1]
        out_list_s, out_list_t = self.get_features(out_list_all)
        loss_transfer = torch.zeros((1,)).cuda()
        if weight_mat is None:
            weight = (1.0 / self.len_seq *
                      torch.ones(self.num_layers, self.len_seq)).cuda()
            # print(weight)
        else:
            weight = weight_mat
            # print(weight)
        dist_mat = torch.zeros(self.num_layers, self.len_seq).cuda()
        ## O(N2)
        for i in range(len(out_list_s)):
            criterion_transder = TransferLoss(
                loss_type=self.trans_loss, input_dim=out_list_s[i].shape[2])
            for j in range(self.len_seq):
                loss_trans = criterion_transder.compute(
                    out_list_s[i][:, j, :], out_list_t[i][:, j, :])
                loss_transfer = loss_transfer + weight[i, j] * loss_trans
                dist_mat[i, j] = loss_trans
        return fc_out, loss_transfer, dist_mat, weight
    # def attention_net(self, lstm_output):
    #     # print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)
    #     #print(lstm_output.shape)
    #     #output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * self.layer_size])
    #     output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size*self.num_layers])
    #     #print(output_reshape.size()) #= (squence_length * batch_size, hidden_size*layer_size)
    #     # tanh(H)
    #     attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega).cuda())
    #     # print(attn_tanh.size()) = (squence_length * batch_size, attention_size)
    #     # 张量相乘
    #     attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
    #     # print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)
    #
    #     exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.len_seq])
    #     # print(exps.size()) = (batch_size, squence_length)
    #
    #     alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
    #     # print(alphas.size()) = (batch_size, squence_length)
    #
    #     alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.len_seq, 1])
    #     # print(alphas_reshape.size()) = (batch_size, squence_length, 1)
    #
    #     state = lstm_output.permute(1, 0, 2)
    #     # print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)
    #
    #     attn_output = torch.sum(state * alphas_reshape, 1)
    #     # print(attn_output.size()) = (batch_size, hidden_size*layer_size)
    #     return attn_output
    # For Boosting-based
    def update_weight_Boosting(self, weight_mat, dist_old, dist_new):
        epsilon = 1e-12     ## 1e-12
        dist_old = dist_old.detach()
        dist_new = dist_new.detach()
        ind = dist_new > dist_old + epsilon
        weight_mat[ind] = weight_mat[ind] * \
            (1 + torch.sigmoid(dist_new[ind] - dist_old[ind]))  ## sigmoid(1 + torch.sigmoid(dist_new[ind] - dist_old[ind]))  ## sigmoid
        weight_norm = torch.norm(weight_mat, dim=1, p=1)
        weight_mat = weight_mat / weight_norm.t().unsqueeze(1).repeat(1, self.len_seq)
        return weight_mat

    def predict(self, x):
        out = self.gru_features(x, predict=True)
        #print(out[1])
        fea = out[0]  ## out[0]
        #fea = out[:,-1,]
        if self.use_bottleneck == True:
            fea_bottleneck = self.bottleneck(fea[:, -1, :])
            fc_out = self.fc(fea_bottleneck).squeeze()
            # smoother = KalmanSmoother(component='level_trend', component_noise={'level':0.9, 'trend':0.9})
            # smoother.smooth(fc_out.cpu().numpy())
            # fc_out = smoother.smooth_data
        else:
            #print(fea.shape)
            #fc_out = self.FC(fea.reshape(fea.shape[0],fea.shape[1]*fea.shape[2]))
            #fc_out = self.attention_net(fea)
            fc_out = self.fc_out(fea[:, -1, :]).squeeze()
            #fc_out = self.Linear(fc_out)
            #fc_out = self.Linear(fea)
            #print(fc_out.shape)
        return fc_out

if __name__ == "__main__":
    x = torch.randn(10,200,12).cuda()
    model = AdaRNN(n_input=12,n_output=10,len_seq=200,use_bottleneck=False).cuda()
    fc_out = model(x)
    #fc_out,loss_tranferss,out_list,_ = model.forward_Boosting(x,None)
    print(fc_out)
    print(fc_out.cpu().shape)
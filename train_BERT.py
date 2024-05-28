import argparse
import datetime
import time

import torch
import torch.nn as nn
import numpy as np
import tensorboardX
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader, dataloader
import matplotlib.pyplot as plt
import warnings

# from base.LSTM_attention import lstm_atten
from base.BERT.Bert import BERT
from base.BERT.sEMG_Bert import sEMG_BERT
from base.ModelListGRU import ModelListGRU

# from tst.model import Muti_Trans

warnings.filterwarnings("ignore")
import metric

from base.GRU_attention import ModelCore
from dataloader import dataLoader, dataLoadercross_zjh, dataLoader10_zjh,dataLoader7_zjh
import tensorboardX as tb

import parser
import os

BATCH_SIZE = 512
OUT_PUT = 10
obj = 0

def str2bool(v):
    # print(v)
    if isinstance(v, bool):
        return v
    if str(v).strip().lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif str(v).strip().lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected. Your input:{v} and input type:{type(v)}')
def get_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='BERT')  # AdaRNN
    parser.add_argument('--log_file', type=str, default='crossBERT_miu.log')
    parser.add_argument('--gpu_id', type=int, default='3')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--subframe', default=200, type=int)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--use_se', type=str2bool, default=False)  # 废弃
    args = parser.parse_args()

    return args


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pprint(*text):
#     # print with UTC+8 time
    time = '[' + str(datetime.datetime.utcnow() +
                     datetime.timedelta(hours=8))[:19] + '] -'
    print(time, *text, flush=True)
    if args.log_file is None:
        return
    with open(args.log_file, 'a') as f:
        print(time, *text, flush=True, file=f)


def train(obj, args):
    train_data, test_data = dataLoader7_zjh(obj=obj, BATCH_SIZE=args.batch_size, TIME_STEP=100, WINDOW_SIZE=200,normal='z-zero')
    loss_func = nn.MSELoss()
    loss_fun = nn.L1Loss()
    writer = tb.SummaryWriter(log_dir='./newlog/BERT/' + args.model_name + '_'+'db7')
    # model = ModelCore().cuda()
    # model = ModelListGRU(input_size=12,output_size=10,rnn_model='GRU',hidden_size=128,num_layers=2,dropout=0.3).cuda()
    # model = RNN(input_size=12, output_size=10, rnn_model=args.model_name, hidden_size=128, num_layers=3,
    #             dropout=0.3).cuda()
    model = sEMG_BERT(vocab_size=args.subframe, hidden=args.hidden, feature_dim=1, n_layers=args.num_layers,attn_heads=8, use_se=args.use_se).cuda()
    # model = lstm_atten(output_size=10,hidden_size=128,nums_layers=1,embed_dim=12,sequence_length=200,dropout=0.3)
    num_model = count_parameters(model)
    print('#model params:', num_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    max_test = 0.0
    h_state = None
    train_cc = []
    test_cc = []
    train_loss = []
    test_loss = []
    index = []
    train_time = []
    
    for i in range(200):
        model.train()
        loss_epoch = 0
        lss = 0
        index.append(i)
        total_output = torch.Tensor([]).cuda()
        total_target = torch.Tensor([]).cuda()
        h_state = None
        since = time.time()
        for step, (data, target) in enumerate(train_data):
            b_x, b_y = data, target
            a_x = b_x.permute(0, 1, 2).type(torch.FloatTensor)
            a_y = b_y.permute(0, 1, 2).type(torch.FloatTensor)
            # print(a_x.shape)
            # print(a_y.shape)
            # a_x = np.swapaxes(data, 1, 2)
            # a_x = np.expand_dims(a_x, axis=1)
            # print(a_x.shape)
            # a_x = Variable(torch.Tensor(a_x)).cuda()
            # a_x = data.type(torch.FloatTensor)
            a_y = b_y.permute(0, 1, 2).type(torch.FloatTensor)

            # print(a_y.shape)
            indices = torch.LongTensor([1, 2, 4, 5, 7, 8, 11, 12, 15, 16])
            # indices = torch.LongTensor([2])
            a_y = torch.index_select(a_y[:, 199:200, :], 2, indices).cuda()

            # print(a_y.shape)  ## [64,1,10]
            # a_y = target.permute(0, 1, 2).type(torch.FloatTensor)
            # indices = torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
            # a_y = torch.index_select(a_y[:, 99:100, :], 2, indices).cuda()
            # print(a_x.shape)
            # print(a_y.shape)
            # print(data.shape)
            # print(target.shape)
            a_x = a_x.float()
            output,_ = model(a_x.cuda())
            # if args.model_name == 'LSTM':
            #     for l in range(len(h_state)):
            #         #print(h_state[l].shape)
            #         h_state[l].detach_()
            # else:
            #     h_state.detach_()
            # print(output)
            # print(output.shape)
            # output = output[0][0]
            # print(output.shape)  ##[64,1,12,200]
            # output = np.squeeze(output)
            # output = np.reshape(output,(64,10))
            # print(output.shape)
            # output = output.view(output.shape[1],
            #                                  output.shape[2])  # .cpu().numpy()  # .astype(np.float32)
            #print(output.shape)

            total_output = torch.cat([total_output, output.resize(args.batch_size, OUT_PUT).cuda()])
            total_target = torch.cat([total_target, a_y.resize(args.batch_size, OUT_PUT).cuda()])
            # print(output.shape)
            loss = loss_func(output, a_y.cuda())
            # + loss_fun(torch.squeeze(output), torch.squeeze(a_y.cuda()))
            # print("[Step {:d}] loss is {:.4f}".format(step, loss.item()))
            ##cc_train = metric.pearsonr(total_output[:, step], total_target[:, step])

            loss_epoch += loss.item()
            lss += 1
            optimizer.zero_grad()
            # print(loss.item())
            loss.backward()
            # torch.nn.utils.clip_grad_value_(model.parameters(), 1.)
            optimizer.step()
        # print("Epoch[{:d}|30] Train Loss = {:.4f}".format(i, loss_epoch / lss))
        time_elapsed = (time.time() - since)
        train_time.append(time_elapsed)
        print(time_elapsed)
        train_loss.append(loss_epoch / len(train_data))
        model.eval()
        loss_fun = nn.MSELoss()
        h_state_1 = None
        with torch.no_grad():
            loss_epoc = 0
            total_output_test = torch.Tensor([]).cuda()
            total_target_test = torch.Tensor([]).cuda()
            for step, (data, target) in enumerate(test_data):
                x = torch.Tensor(
                    np.transpose(np.array(data, dtype='float32'), (0, 1, 2)))  # (batchsize,channel,inputsize)
                y = torch.Tensor(
                    np.transpose(np.array(target, dtype='float32'), (0, 1, 2)))  # (batchsize,inputsize,channel,)
                # x = np.expand_dims(x,axis=1)
                # print(x.shape)
                x = Variable(torch.Tensor(x)).cuda()
                indices = torch.LongTensor([1, 2, 4, 5, 7, 8, 11, 12, 15, 16])
                # indices = torch.LongTensor([2])
                y = torch.index_select(y[:, 199:200, :], 2, indices).cuda()
                output_test,_ = model(x.cuda())
                # output_test = output_test.view(output_test.shape[1],
                #                              output_test.shape[2])  # .cpu().numpy()  # .astype(np.float32)

                # output_test = output_test[0][0]
                # print(output_test)
                total_output_test = torch.cat([total_output_test, output_test.view([args.batch_size, OUT_PUT])])
                total_target_test = torch.cat([total_target_test, y.view([args.batch_size, OUT_PUT]).cuda()])
                # loss_tes = loss_func(torch.squeeze(output_test), torch.squeeze(y.cuda()))
                loss_tes = loss_func(output_test, y.cuda())
                # print("[Step {:d}] loss is {:.4f}".format(step, loss.item()))
                ##cc_train = metric.pearsonr(total_output[:, step], total_target[:, step])
                loss_epoc += loss_tes.item()
            print("Epoch[{:d}|300] Test Loss = {:.4f}".format(i+1, loss_epoch / lss))
            # if step >= 29:
            #     torch.save(model, args.model_name + "1021sub" + str(1) + '_kernel' + str(3) + '.pkl')
            pprint("Epoch[{:d}|400] Train Loss = {:.4f} Test Loss = {:.4f}".format(i+1, loss_epoch / lss, loss_epoc / len(test_data)))
            test_loss.append(loss_epoc / len(test_data))
            aver_test = 0
            aver_train = 0
            aver_mse = 0.0
            aver_rmse = 0.0
            aver_r = 0.0
            for id in range(10):
                cc_train = metric.pearsonr(total_output[:, id], total_target[:, id])
                cc_test = metric.pearsonr(total_output_test[:, id], total_target_test[:, id])
                aver_test = aver_test + cc_test
                aver_train = aver_train + cc_train
                mse = metric.MSE(total_output_test[:, id], total_target_test[:, id])
                rmse = metric.RMSE(total_output_test[:, id], total_target_test[:, id])
                r = metric.R_Squared(total_output_test[:, id].cpu().numpy(), total_target_test[:, id].cpu().numpy())
                aver_mse += mse
                aver_rmse += rmse
                aver_r += r
                print(id, " ", cc_test)
            aver_test = aver_test / 10.0
            aver_train = aver_train / 10.0
            aver_mse = aver_mse / 10.0
            aver_r = aver_r / 10.0
            aver_rmse = aver_rmse / 10.0
            # pprint(aver_train, aver_test)
            pprint("Epoch ", i + 1, "Test CC:", aver_test.cpu(), "MSE:", aver_mse, "RMSE:", aver_rmse, "R2:", aver_r)
            train_cc.append(aver_train.item())
            test_cc.append(aver_test.item())
            writer.add_scalar('train_loss', loss_epoch / lss, i)
            writer.add_scalar('test_loss', loss_epoc / len(test_data), i)
            writer.add_scalar('Train_CC', aver_train, i)
            writer.add_scalar('Test_CC', aver_test, i)
        if (max_test < aver_test):
        #     max_test = aver_test
        #     # fig = plt.figure(figsize=(20, 10))
        #     # for i in range(10):
        #     #     plt.subplot(5, 2, i + 1)
        #     #     plt.plot(total_output_test[:, i].detach().cpu().numpy())
        #     #     plt.plot(total_target_test[:, i].detach().cpu().numpy())
        #     # fig.savefig("result_GRU_Self_" + str(obj))
            torch.save(model,
                       './save/new/DB7/BERT_EMSA/' + '' +args.model_name + "_sub" + str(obj) + '_' + str(args.batch_size) +  'z-zero_3_0.3' + '.pkl')
            # torch.save(model.state_dict(), './save/TEST.pth')
        print(max_test)
    
    # pprint(time_elapsed)
    print('Training complete in {:.3f}m {:.3f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(sum(train_time)/len(train_time))
    # plt.title('RNN Loss')
    # plt.plot(index, train_loss, color='blue', label='train_loss')
    # plt.plot(index, test_loss, color='red', label='test_loss')
    # plt.legend()
    # plt.savefig("result_RNN_Loss")
    # plt.show()
    # plt.title('RNN CC')
    # plt.plot(index, train_cc, color='blue', label='train_cc')
    # plt.plot(index, test_cc, color='red', label='test_cc')
    # plt.legend()
    # plt.savefig("result_RNN_CC")
    # plt.show()


if __name__ == '__main__':
    # obj = [1, 5, 13, 14, 17, 19, 23, 33]
    #obj = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    obj = [1,2,3,4,5,6,7,8,9,10]
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    # train(obj=5,args=args)
    # train(obj)
    for i in range(len(obj)):
        train(obj=obj[i], args=args)

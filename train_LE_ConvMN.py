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
from base.ModelListGRU import ModelListGRU
from data_processing import ninapro_12_200
# from tst.model import Muti_Trans

warnings.filterwarnings("ignore")
import metric
from base.LE_ConvMN import LE_convMN
from base.GRU_attention import ModelCore
from dataloader import dataLoader, dataLoader_zph, dataLoader_zjh,dataLoader10_zjh
import tensorboardX as tb

import parser
import os

BATCH_SIZE = 1
OUT_PUT = 10
obj = 5


def get_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='LE-ConvMN')  # AdaRNN
    parser.add_argument('--log_file', type=str, default='test.log')
    parser.add_argument('--gpu_id', type=int, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--obj",type=int,default=1)

    args = parser.parse_args()

    return args


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pprint(*text):
    # print with UTC+8 time
    time = '[' + str(datetime.datetime.utcnow() +
                     datetime.timedelta(hours=8))[:19] + '] -'
    print(time, *text, flush=True)
    if args.log_file is None:
        return
    with open(args.log_file, 'a') as f:
        print(time, *text, flush=True, file=f)


def train(obj, args):
    train_data, test_data = dataLoader10_zjh(obj=obj, BATCH_SIZE=args.batch_size, TIME_STEP=100, WINDOW_SIZE=200,normal='miu')
    
    loss_func = nn.MSELoss()
    loss_func = loss_func.cuda()
    loss_fun = nn.L1Loss()
    # writer = tb.SummaryWriter(log_dir='./newlog/LE-Conv_MN/' + args.model_name + '0206')
    # model = ModelCore().cuda()
    # model = ModelListGRU(input_size=12,output_size=10,rnn_model='GRU',hidden_size=128,num_layers=2,dropout=0.3).cuda()
    model = LE_convMN((12, 200), 1, 1, [64, 32, 10], (3, 3), 3, 20, batch_first=True, bias=True,
                       return_all_layers=False, withCBAM=False).cuda()
#    model_str = './save/new/LE-Conv_MN/LE-ConvMNsub'+str(obj) + '_1miu_3_0.3.pkl'
#    model.load_state_dict(torch.load(model_str))
    # torch.distributed.init_process_group(backend="nccl")
    # local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(local_rank)
    # device = torch.device("cuda",local_rank)
    # model.cuda(local_rank)
    # print("Let's use", torch.cuda.device_count(), "GPUs!")
    # # 5) 封装
    # model = torch.nn.parallel.DistributedDataParallel(model,
    #                                                 device_ids=[local_rank],
    #                                                 output_device=local_rank,find_unused_parameters=True)
    
    # TIME_STEP = 100
    # WINDOW_SIZE = 200
    # normal = "miu"
    # data_read = ninapro_12_200.NINAPRO_1('./data/ninapro10_40' + '/S' + str(obj) + '_E2_A_rms_train.csv',
    #                                      './data/ninapro10_40' + '/S' + str(obj) + '_E2_A_glove_train.csv',
    #                                      './data/ninapro10_40' + '/S' + str(obj) + '_E2_A_rms_test.csv',
    #                                      './data/ninapro10_40' + '/S' + str(obj) + '_E2_A_glove_test.csv', TIME_STEP,
    #                                      WINDOW_SIZE,
    #                                      Normalization = normal,
    #                                      train=True)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(data_read)
    # train_data = DataLoader(dataset=data_read, batch_size=BATCH_SIZE, shuffle=False, drop_last=True,num_workers=8,sampler=train_sampler,pin_memory=True)

    # data_read_test = ninapro_12_200.NINAPRO_1('./data/ninapro10_40' + '/S' + str(obj) + '_E2_A_rms_train.csv',
    #                                           './data/ninapro10_40' + '/S' + str(obj) + '_E2_A_glove_train.csv',
    #                                           './data/ninapro10_40' + '/S' + str(obj) + '_E2_A_rms_test.csv',
    #                                           './data/ninapro10_40' + '/S' + str(obj) + '_E2_A_glove_test.csv', TIME_STEP,
    #                                           WINDOW_SIZE,
    #                                           Normalization = normal,
    #                                           train=False)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(data_read_test)
    # test_data = DataLoader(dataset=data_read_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=True,num_workers=8,sampler=test_sampler,pin_memory=True)
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
    
    for i in range(1500):
        model.train()
        loss_epoch = 0
        lss = 0
        index.append(i)
        total_output = torch.Tensor([]).cuda()
        total_target = torch.Tensor([]).cuda()
        hidden = None
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
            # a_y = b_y.permute(0, 1, 2).type(torch.FloatTensor)

            # print(a_y.shape)
            indices = torch.LongTensor([1, 2, 4, 5, 7, 8, 11, 12, 15, 16])
            a_y = torch.index_select(a_y[:, 199:200, :], 2, indices).cuda()
            a_x = torch.unsqueeze(a_x,3)
            output, hidden = model(a_x.cuda(), hidden) # LE-ConvMN
            for l in range(len(hidden)):
                for p in range(len(hidden[l])):
                    hidden[l][p].detach_()
            # a_x = a_x.float()
            # output = model(a_x.cuda())
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
            total_output = torch.cat([total_output, output.resize(args.batch_size, OUT_PUT).cuda()])
            total_target = torch.cat([total_target, a_y.resize(args.batch_size, OUT_PUT).cuda()])
            # print(output.shape)
            loss = loss_func(output, a_y.cuda())
            # loss = loss_func(torch.squeeze(output), torch.squeeze(a_y.cuda()))
            # + loss_fun(torch.squeeze(output), torch.squeeze(a_y.cuda()))
            # print("[Step {:d}] loss is {:.4f}".format(step, loss.item()))
            ##cc_train = metric.pearsonr(total_output[:, step], total_target[:, step])
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            loss_epoch += loss.item()
            lss += 1
            optimizer.zero_grad()
            # print(loss.item())
        print(time.time() - since)    
            
            
        # print("Epoch[{:d}|30] Train Loss = {:.4f}".format(i, loss_epoch / lss))
        train_loss.append(loss_epoch / len(train_data))
        model.eval()
        loss_fun = nn.MSELoss()
        h_state = None
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
                # output_test = model(x.cuda())
                x = torch.unsqueeze(x, 3)
                output_test, h_state = model(x.cuda(), h_state)
                # output_test = output_test[0][0]
                # print(output_test)
                total_output_test = torch.cat([total_output_test, output_test.view([args.batch_size, OUT_PUT])])
                total_target_test = torch.cat([total_target_test, y.view([args.batch_size, OUT_PUT]).cuda()])
                loss_tes = loss_func(output_test, y.cuda())
                # print("[Step {:d}] loss is {:.4f}".format(step, loss.item()))
                ##cc_train = metric.pearsonr(total_output[:, step], total_target[:, step])
                loss_epoc += loss_tes.item()
            print("Epoch[{:d}|800] Test Loss = {:.4f}".format(i+1, loss_epoch / lss))
            # if step >= 29:
            #     torch.save(model, args.model_name + "1108sub" + str(1) + '_kernel' + str(3) + '.pkl')
            pprint("Epoch[{:d}|800] Train Loss = {:.4f} Test Loss = {:.4f}".format(i+1, loss_epoch / lss,
                                                                                   loss_epoc / len(test_data)))
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
            pprint(aver_train, aver_test)
            pprint("Epoch ", i + 1, "Test CC:", aver_test.cpu(), "MSE:", aver_mse, "RMSE:", aver_rmse, "R2:",
                   aver_r)
            train_cc.append(aver_train.item())
            test_cc.append(aver_test.item())
            # writer.add_scalar('train_loss', loss_epoch / lss, i)
            # writer.add_scalar('test_loss', loss_epoc / len(test_data), i)
            # writer.add_scalar('Train_CC', aver_train, i)
            # writer.add_scalar('Test_CC', aver_test, i)
        if (max_test < aver_test):
            max_test = aver_test
            # fig = plt.figure(figsize=(20, 10))
            # for i in range(10):
            #     plt.subplot(5, 2, i + 1)
            #     plt.plot(total_output_test[:, i].detach().cpu().numpy())
            #     plt.plot(total_target_test[:, i].detach().cpu().numpy())
            # fig.savefig("result_GRU_Self_" + str(obj))
            #torch.save(model,
                       # './save/new/LE-Conv_MN/' + args.model_name + "miusub" + str(obj) + '_' + str(args.batch_size) + '_3_0.3' + '.pkl')
            # torch.save(model.state_dict(), './save/TEST.pth')
        pprint(max_test)
        print(max_test)
    time_elapsed = (time.time() - since) / 300
    pprint(time_elapsed)
    pprint('Training complete in {:.3f}m {:.3f}s'.format(time_elapsed // 60, time_elapsed % 60))
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
    obj = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
    # obj = [1]
    args = get_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    # train(obj=5,args=args)
    train(obj = args.obj,args = args)
#    for i in range(len(obj)):
#        train(obj=obj[i], args=args)

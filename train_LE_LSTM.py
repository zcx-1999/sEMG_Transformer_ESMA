import argparse
import datetime
import time

import torch
import torch.nn as nn
import numpy as np
import tensorboardX
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import warnings

from base.LE_RNN import RNN


warnings.filterwarnings("ignore")
import metric


from dataloader import dataLoader, dataLoader_zph, dataLoader10_zjh,dataLoadercross_zjh
import tensorboardX as tb
import parser
import os

BATCH_SIZE = 1
OUT_PUT = 10

obj = 5
def get_args():

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='LE-LSTM') #AdaRNN
    parser.add_argument('--log_file', type=str, default='test.log')
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--input',type=int,default=12)
    parser.add_argument('--out_put',type=int,default=10)
    parser.add_argument('--gpu_id',type=int,default='2')
    parser.add_argument('--obj',type=int,default='0')


    args = parser.parse_args()

    return args
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(obj):
    train_data,test_data = dataLoadercross_zjh(obj=obj,BATCH_SIZE=args.batch_size,TIME_STEP=200, WINDOW_SIZE=200,normal='miu')
    loss_func = nn.MSELoss()
    loss_fun = nn.L1Loss()
    writer = tb.SummaryWriter(log_dir='./newlog/LE-LSTM/' +args.model_name+'_0111' )

    #model = ModelCore().cuda()
    #model = ModelListGRU(input_size=12,output_size=10,rnn_model='GRU',hidden_size=128,num_layers=2,dropout=0.3).cuda()
    model = RNN(input_size=args.input,output_size=args.out_put,rnn_model=args.model_name,hidden_size=128,num_layers=3,dropout=0.3).cuda()
    #model = lstm_atten(output_size=10,hidden_size=128,nums_layers=1,embed_dim=12,sequence_length=200,dropout=0.3)
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
    since = time.time()
    h_state = None
    for i in range(400):
        model.train()
        loss_epoch = 0
        lss = 0
        index.append(i)
        
        total_output = torch.Tensor([]).cuda()
        total_target = torch.Tensor([]).cuda()
        for step, (data,target) in enumerate(train_data):
            
            b_x, b_y = data, target
            a_x = b_x.permute(0, 1, 2).type(torch.FloatTensor)
            a_y = b_y.permute(0, 1, 2).type(torch.FloatTensor)
            #print(a_x.shape)
            #print(a_y.shape)
            #a_x = np.swapaxes(data, 1, 2)
            #a_x = np.expand_dims(a_x, axis=1)
            #print(a_x.shape)
            #a_x = Variable(torch.Tensor(a_x)).cuda()
            #a_x = data.type(torch.FloatTensor)
            a_y = b_y.permute(0, 1, 2).type(torch.FloatTensor)

            #print(a_y.shape)
            indices = torch.LongTensor([1, 2, 4, 5, 7, 8, 11, 12, 15, 16])
            #indices = torch.LongTensor([2])
            a_y = torch.index_select(a_y[:, 199:200, :], 2, indices).cuda()

            #print(a_y.shape)  ## [64,1,10]
            #a_y = target.permute(0, 1, 2).type(torch.FloatTensor)
            #indices = torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
            #a_y = torch.index_select(a_y[:, 99:100, :], 2, indices).cuda()
            #print(a_x.shape)
            #print(a_y.shape)
            # print(data.shape)
            # print(target.shape)
            a_x = a_x.float()
            output,h_state = model(a_x.cuda(),h_state)
            if args.model_name == 'LE-LSTM':
                for l in range(len(h_state)):
                #print(h_state[l].shape)
                    h_state[l].detach_()
            else:
                h_state.detach_()
            # print(output)
            #print(output.shape)
            # output = output[0][0]
            #print(output.shape)  ##[64,1,12,200]
            #output = np.squeeze(output)
            #output = np.reshape(output,(64,10))
            #print(output.shape)
            total_output = torch.cat([total_output, output.resize(args.batch_size, args.out_put).cuda()])
            total_target = torch.cat([total_target, a_y.resize(args.batch_size, args.out_put).cuda()])
            #print(output.shape)
            # loss = loss_func(torch.squeeze(output), torch.squeeze(a_y.cuda()))
            loss = loss_func(output, a_y.cuda())
            # + loss_fun(torch.squeeze(output), torch.squeeze(a_y.cuda()))
            # print("[Step {:d}] loss is {:.4f}".format(step, loss.item()))
            ##cc_train = metric.pearsonr(total_output[:, step], total_target[:, step])

            loss_epoch += loss.item()
            lss+=1

            #print(loss.item())
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ##torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        time_elapsed = (time.time() - since)
        print(time_elapsed)
            
        # print("Epoch[{:d}|30] Train Loss = {:.4f}".format(i, loss_epoch / lss))
        train_loss.append(loss_epoch/len(train_data))
        model.eval()
        loss_fun = nn.MSELoss()
        h_state_1 = None
        with torch.no_grad():
            loss_epoc = 0
            total_output_test = torch.Tensor([]).cuda()
            total_target_test = torch.Tensor([]).cuda()
            for step, (data, target) in enumerate(test_data):
                x = torch.Tensor(np.transpose(np.array(data, dtype='float32'), (0, 1, 2)))  # (batchsize,channel,inputsize)
                y = torch.Tensor(np.transpose(np.array(target, dtype='float32'), (0, 1, 2)))  # (batchsize,inputsize,channel,)
                #x = np.expand_dims(x,axis=1)
                #print(x.shape)
                x = Variable(torch.Tensor(x)).cuda()
                indices = torch.LongTensor([1, 2, 4, 5, 7, 8, 11, 12, 15, 16])
                #indices = torch.LongTensor([2])
                y = torch.index_select(y[:, 199:200, :], 2, indices).cuda()
                output_test,h_state_1 = model(x.cuda(),h_state_1)
                #output_test = output_test[0][0]
                #print(output_test)
                total_output_test = torch.cat([total_output_test, output_test.view([args.batch_size, args.out_put])])
                total_target_test = torch.cat([total_target_test, y.view([args.batch_size, args.out_put]).cuda()])
                loss_tes = loss_fun(output_test, y.cuda()) 
                # + loss_fun(torch.squeeze(output), torch.squeeze(a_y.cuda()))
                # print("[Step {:d}] loss is {:.4f}".format(step, loss.item()))
                ##cc_train = metric.pearsonr(total_output[:, step], total_target[:, step])
                loss_epoc += loss_tes.item()
            #print("Epoch[{:d}|30] Test Loss = {:.4f}".format(i, loss_epoch / lss))
            # if step >= 29:
            #     torch.save(model, 'Conv_LSTM' + "sub" + str(1) + '_kernel' + str(3) + '.pkl')
            print("Epoch[{:d}|300] Train Loss = {:.4f} Test Loss = {:.4f}".format(i, loss_epoch / lss,loss_epoc/len(test_data)))
            test_loss.append(loss_epoc/len(test_data))
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
                print(id," ",cc_test)
            aver_test = aver_test / 10.0
            aver_train = aver_train / 10.0
            aver_mse = aver_mse / 10.0
            aver_r = aver_r / 10.0
            aver_rmse = aver_rmse / 10.0
            print(aver_train,aver_test)
            print("Epoch ", i + 1, "Test CC:", aver_test.cpu(), "MSE:", aver_mse, "RMSE:", aver_rmse, "R2:",
                  aver_r)
            train_cc.append(aver_train.item())
            test_cc.append(aver_test.item())
            # writer.add_scalar('train_loss',loss_epoch/lss,i)
            # writer.add_scalar('test_loss',loss_epoc/len(test_data),i)
            # writer.add_scalar('Train_CC',cc_train,i)
            # writer.add_scalar('Test_CC',cc_test,i)
        if (max_test < aver_test):
            max_test = aver_test
            # fig = plt.figure(figsize=(20, 10))
            # for i in range(10):
            #     plt.subplot(5, 2, i + 1)
            #     plt.plot(total_output_test[:, i].detach().cpu().numpy())
            #     plt.plot(total_target_test[:, i].detach().cpu().numpy())
            # fig.savefig("result_GRU_Self_" + str(obj))
            # torch.save(model, './save/new/LE-LSTM/'+args.model_name+'miu__' + "sub" + str(obj) + "_" + str(args.batch_size) + '_3_0.3' + '.pkl')
            # torch.save(model.state_dict(), './save/TEST.pth')
        print(max_test)

    print('Training complete in {:.3f}m {:.3f}s'.format(time_elapsed // 60, time_elapsed % 60))
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



if __name__ =='__main__':
    # obj = [1, 5, 13, 14, 17, 19, 23, 33]
    # obj = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
    #obj = [1, 5, 7, 9, 17, 19, 23, 24]
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    train(obj=args.obj)
    #main_transfer(args,args.obj)
    #train(obj)
    # for i in range(len(obj)):
        # train(obj=obj[i])

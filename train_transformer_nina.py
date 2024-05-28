import torch.nn as nn
import torch
import torch.optim as optim

import os
import argparse
import datetime
import numpy as np
import metric
from dataloader import dataLoader_zph,dataLoader_zjh,dataLoader23_zjh,dataLoader10_zjh,dataLoader7_zjh,dataLoader231_zjh,dataLoadercross_zjh
from utils import utils

import matplotlib.pyplot as plt
from tst import Transformer
import warnings
import time
import tensorboardX as tb
from tensorboardX import SummaryWriter
from base.loss.LabelSmoothing import LabelSmoothing
from tst.utils import *
from base.loss.Balanced_MSE import BMCLossMD
#from base.loss.sdtw_loss import SoftDTW
from base.loss.soft_dtw import SoftDTW
from base.loss.diliate_loss.dilate_loss import dilate_loss
warnings.filterwarnings("ignore")


# def pprint(*text):
#     # print with UTC+8 time
#     time = '[' + str(datetime.datetime.utcnow() +
#                      datetime.timedelta(hours=8))[:19] + '] -'
#     print(time, *text, flush=True)
#     if args.log_file is None:
#         return
#     with open(args.log_file, 'a') as f:
#         print(time, *text, flush=True, file=f)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(args, model, optimizer, data, epoch,ffn_indicators=None):
    model.train()
    
    criterion_1 = nn.L1Loss()
    criterion_BMSE = BMCLossMD(init_noise_sigma=0.0001).cuda()
    criterion_DWT = SoftDTW(gamma=1.0,normalize=False)
    total_output = torch.Tensor([]).cuda()
    total_target = torch.Tensor([]).cuda()
    criterion_2 = nn.MSELoss().cuda()
    criterion_smothing = LabelSmoothing(smoothing=0.3)
    if args.loss_type == 'MSE':
        criterion = nn.MSELoss()
    elif args.loss_type == 'DWT':
        criterion = SoftDTW(gamma=1.0,normalize=False)
    else:
        print("没有此损失函数")
    loss_sum = 0.0
    since = time.time()
    for i,(fea,label) in enumerate(data):
        optimizer.zero_grad()
        a_x = fea.type(torch.FloatTensor).cuda()
        b_x = label.type(torch.FloatTensor).cuda()
        a_y = b_x.permute(0, 1, 2).type(torch.FloatTensor)
        indices = torch.LongTensor([1, 2, 4, 5, 7, 8, 11, 12, 15, 16])
        a_y = torch.index_select(a_y[:, 199:200, :], 2, indices).cpu()
        fea = fea.float()
        pred_all, ffn_indicators = model(fea.cuda())
        total_output = torch.cat([total_output, pred_all.resize(args.batch_size, args.class_num)])
        total_target = torch.cat([total_target, a_y.resize(args.batch_size, args.class_num).cuda()])
        #pred_all = pred_all.t()
        a_y = a_y.resize(args.batch_size, args.class_num)
        #a_y = a_y.t()
        #print(a_y.shape)
        #print(pred_all.shape)
        #a_y = torch.squeeze(a_y,dim=1)
        #pred_all = torch.unsqueeze(torch.squeeze(pred_all),0)
        #a_y = torch.unsqueeze(torch.squeeze(a_y),0)
        # loss = criterion_DWT(pred_all,a_y.cuda())
        # loss = criterion_DWT(pred_all, a_y.cuda())/100000
        #loss = dilate_loss(outputs=pre,targets=a_.cuda(),alpha=0.5,gamma=0.001,# device='cuda')
        # loss = criterion_smothing(torch.squeeze(pred_all), torch.squeeze(a_y).cuda())
        # loss = criterion_2(pred_all,a_y.cuda())
        loss = criterion(pred_all,a_y.cuda())
        loss_sum += loss.item()
        optimizer.zero_grad()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        loss.backward()
        optimizer.step()
    #time_train.append(time.time() - since)
    print(time.time() - since)
    aver_train = 0
    for id in range(10):
        cc_train = metric.pearsonr(total_output[:, id], total_target[:, id])
        aver_train = aver_train + cc_train
    aver_train = aver_train / 10.0
    print(epoch+1,"Train LOSS",loss_sum / len(data))
    print("Epoch ", epoch+1,"Train CC",aver_train)
    return loss_sum / len(data), aver_train, ffn_indicators


def p_test_epoch(args,obj,model,Eoach, data,max_test = 0):
    model.eval()
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    criterion_smothing = LabelSmoothing(smoothing=0.3)
    criterion_BMSE = BMCLossMD(init_noise_sigma=0.1).cuda()
    criterion_DWT = SoftDTW(gamma=1.0,normalize=False)
    total_output = torch.Tensor([]).cuda()
    total_target = torch.Tensor([]).cuda()
    if args.loss_type == 'MSE':
        criterion = nn.MSELoss()
    elif args.loss_type == 'DWT':
        criterion = SoftDTW(gamma=1.0,normalize=False)
    else:
        print("没有此损失函数")
    loss_sum = 0.0
    data_sum = len(data)

    with torch.no_grad():
        for i, (data, target) in enumerate(data):
            a_x = data.type(torch.FloatTensor).cuda()
            b_x = target.type(torch.FloatTensor).cuda()
            b_x, b_y = data, target
            a_x = b_x.permute(0, 1, 2).type(torch.FloatTensor)
            a_y = b_y.permute(0, 1, 2).type(torch.FloatTensor)
            indices = torch.LongTensor([1, 2, 4, 5, 7, 8, 11, 12, 15, 16])
            a_y = torch.index_select(a_y[:, 199:200, :], 2, indices).cpu()
            pred, _ = model(a_x.cuda())
            a_y = a_y.resize(args.batch_size, args.class_num)
            #pred = pred.t()
            #a_y = a_y.t()
            #a_y = torch.squeeze(a_y,dim=1)
            total_output = torch.cat([total_output, pred.resize(args.batch_size, args.class_num)])
            total_target = torch.cat([total_target, a_y.resize(args.batch_size, args.class_num).cuda()])
            #a_y = torch.unsqueeze(torch.squeeze(a_y),2)
            # pred = torch.unsqueeze(torch.squeeze(pred),2)
            #loss = criterion_DWT(pred,a_y.cuda())
            #loss = dilate_loss(outputs=pred,targets=a_y.cuda(),alpha=0.5,gamma=0.001,device='cuda')
           
            loss = criterion(pred, a_y.cuda())
            loss_sum += loss.item()
    aver_train = 0
    #
    aver_mse = 0.0
    aver_rmse = 0.0
    aver_r = 0.0
    # if args.smooth_label:
    # print(total_output.shape)
    total_output = avg_smoothing_np(3, total_output)
    for id in range(10):
        cc_train = metric.pearsonr(total_output[:, id], total_target[:, id])
        mse = metric.MSE(total_output[:, id], total_target[:, id])
        rmse = metric.RMSE(total_output[:, id], total_target[:, id])
        r = metric.R_Squared(total_output[:, id].cpu().numpy(), total_target[:, id].cpu().numpy())
        aver_mse += mse
        aver_rmse += rmse
        aver_r += r
        aver_train = aver_train + cc_train
    aver_train = aver_train / 10.0
    aver_mse = aver_mse / 10.0
    aver_r = aver_r / 10.0
    aver_rmse = aver_rmse / 10.0
    if aver_train > max_test:
        #if (Eoach > 1):
            # fig = plt.figure(figsize=(20, 10))
            # for i in range(10):
            #     plt.subplot(5, 2, i + 1)
            #     plt.plot(total_output[:, i].detach().cpu().numpy())
            #     plt.plot(total_target[:, i].detach().cpu().numpy())
            # fig.savefig("transformer_tiaocan" + str(obj))
            # torch.save(model, './my_pkl/Transformer_EMSA_sqe_lays3_DWT' +  "_sub_"  + str(obj) + '_batch' + str(args.batch_size) + args.normal + '.pkl')
            # torch.save(model,'./my_pkl/ZJH_40_Zero/Soft_dwt/Transformer_EMSA_lays3_DWT'+'_sub'+str(obj)+'_batch'+str(args.batch_size) + args.normal + '.pkl')
            # torch.save(model,'./my_pkl/ZJH_40_Zero/MSE/Transformer_EMSA_lays3_MSE'+'_sub'+str(obj)+'_batch'+str(args.batch_size) + args.normal + '.pkl')
            #torch.save(model,'./save/new/myTransformer/Cross_subject_Transforme200200_lays'+str(args.num_layer)+'_DTW'+ '_' +str(args.hidden_dim) +'_sub'+str(obj)+'_batch'+str(args.batch_size) + args.normal + '.pkl')
            torch.save(model,'./save/new/DB7/Transformer/Transformer_lays'+str(args.num_layer)+'_DTW'+ '_' +str(args.hidden_dim) +'_sub'+str(obj)+'_batch'+str(args.batch_size) + args.normal + '.pkl')
            #torch.save(model.state_dict(), './save/TEST.pth')
            #print(aver_train)
    print(Eoach + 1, "Test LOSS", loss_sum / data_sum)
    print("Epoch ", Eoach + 1, "Test CC:", aver_train.cpu(), "MSE:", aver_mse, "RMSE:", aver_rmse, "R2:", aver_r)
    return loss_sum / data_sum,aver_train


def main_transfer(args,OBJ):
    print(args)

    # output_path = args.outdir + args.database + '_' + args.model_name + '_S'+ str(OBJ) + '_' + \
                  # args.loss_type + '_' + str(args.pre_epoch) + \
                  # '_' + '_' + str(args.lr) + "_"  + "-layer-num-" + str(
        # args.num_layer) + "-hidden-" + str(args.hidden_dim) + "-num_head-" + str(args.num_head) + "dw-" + str(args.dw)
    # "-hidden" + str(args.hidden_dim) + "-head" + str(args.num_head)
    # save_model_name = args.model_name + '_' + args.loss_type + \
                    #   '_' + str(args.dw) + '_' + str(args.lr) + '.pkl'
    # utils.dir_exist(output_path)
    # pprint('create loaders...')
    writer = tb.SummaryWriter(log_dir='./log/db7/')
    # print(output_path)
    train_data, test_data = dataLoader7_zjh(obj=OBJ, BATCH_SIZE=args.batch_size, TIME_STEP=args.Time_Step, WINDOW_SIZE=args.Windows,normal=args.normal)

    # args.log_file = os.path.join(output_path, '0226'+str(OBJ)+'run.log')
    print('create model...')
    ######
    # Model parameters
    d_model = args.hidden_dim  # 32  Lattent dim
    q = 16  # Query size  8
    v = 16  # Value size  8
    h = args.num_head  # 4   Number of heads
    N = args.num_layer  # Number of encoder and decoder to stack
    attention_size = 16 # Attention window size #8
    pe = "regular" #"regular"  # Positional encoding     "regular"
    chunk_mode = "EMSA"# "window" # "EMSA" #"window" # "EMSA"#"chunk" #"chunk" #"window" # "window" ## "chunk"  "window"
    d_input = 12  # From dataset
    d_output = 10  # From dataset

    # model1 = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size, chunk_mode=chunk_mode,
                        # pe=pe,Unified=True, pe_period=0).cuda()
    # optimizer = optim.Adam(model1.parameters(), lr=args.lr)
    ffn_indicators = None
    # for i in range(args.pre_epoch):
        # loss_train, aver_train, ffn_indicators = train_epoch(
           # args, model1, optimizer, train_data, i,  ffn_indicators)
    #print(ffn_indicators.sum())
    model = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size, chunk_mode=chunk_mode,
                        pe=pe, pe_period=0,Unified=False,ffn_indicators=ffn_indicators).cuda()

    #####
    num_model = count_parameters(model)
    print('#model params:', num_model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    best_score = np.inf
    best_epoch, stop_round = 0, 0
    max_test = -1001
    
    train_cc = []
    test_cc = []
    train_loss = []
    test_loss = []
    index = []
    time_train = []
    for epoch in range(args.n_epochs):

        print('Epoch:', epoch)
        print('training...')
        index.append(epoch)
        since = time.time()
        loss_train, aver_train, ffn_indicators = train_epoch(
            args, model, optimizer, train_data, epoch, ffn_indicators)
        time_train.append(time.time() - since)
        print(time.time() - since)
        train_loss.append(loss_train)
        train_cc.append(aver_train.item())
        print('evaluating...')
        loss_test,te = p_test_epoch(args,OBJ, model, epoch, test_data, max_test=max_test)
        test_loss.append(loss_test)
        test_cc.append(te.item())
        if te > max_test:
            max_test = te
        # scheduler.step()
        # pprint("Max_CC", max_test)
        writer.add_scalar('train_loss',loss_train,epoch)
        writer.add_scalar('test_loss',loss_test,epoch)
        writer.add_scalar('Train_CC',aver_train.item(),epoch)
        writer.add_scalar('Test_CC',te.item(),epoch)
    # pprint((time.time() - since))
    # time_elapsed = (time.time() - since) / args.n_epochs
    time_elapsed = sum(time_train)/len(time_train)
    print(time_elapsed)
    print('Training complete in {:.3f}m {:.3f}s {:.3f}s'.format(time_elapsed // 60, time_elapsed % 60,time_elapsed))
    # plt.title('Transformer Loss')
    # plt.plot(index, train_loss, color='blue', label='train_loss')
    # plt.plot(index, test_loss, color='red', label='test_loss')
    # plt.legend()
    # plt.savefig("result_Transformer_Loss")
    # plt.show()
    # plt.title('Transformer CC')
    # plt.plot(index, train_cc, color='blue', label='train_cc')
    # plt.plot(index, test_cc, color='red', label='test_cc')
    # plt.legend()
    # plt.savefig("result_Transformer_CC")
    # plt.show()
    return max_test


def get_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='Transformer_EMSA')
    parser.add_argument('--d_feat', type=int, default=12)
    parser.add_argument('--normal',type=str,default="z-zero") ## z-zero
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--class_num', type=int, default=10)
    parser.add_argument('--pre_epoch', type=int, default=10)  # 10
    parser.add_argument('--num_layer', type=int, default=2)  # 25

    # training
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)  ## 0.0001
    parser.add_argument('--early_stop', type=int, default=40)
    parser.add_argument('--smooth_steps', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dw', type=float, default=1.0)
    parser.add_argument('--loss_type', type=str, default='MSE')
    parser.add_argument('--database', type=str, default='Ninapro')
    parser.add_argument('--hidden_dim', type=int, default=128)#64
    parser.add_argument('--num_head', type=int, default=8)  ## 8
    parser.add_argument('--smooth_label',type=bool,default=False)

    # other
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--data_path', default="./data")
    parser.add_argument('--outdir', default='./new_log/Transformer/')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--log_file', type=str, default='Transformer_EMSA_no.log')
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--obj', type=int, default=1)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--Time_Step', type=int, default=100)
    parser.add_argument('--Windows', type=int, default=200)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # obj = [1, 5, 13, 14, 17, 19, 23, 33]
    #obj = [1, 5, 7, 9, 17, 19, 23, 24]
    # obj = [6,7,14,16,19,20,24,26,31,33,37]
    obj = [1,2,3,4,5,6,7,8,9,10]
    #obj = [1,3,4,5,8,14,16,18,19,20,22,23,24,26,28,35]
    # obj = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
    # main_transfer(args,args.obj)
    list = []
    for i in range(len(obj)):
        cc = main_transfer(args,OBJ=obj[i])
        list.append(cc)
        # pprint("这是第", obj[i], "个人的数据：")
    for i in range(len(obj)):
        print("这是第",obj[i],"个人的数据:,CC为",list[i])

import argparse

import torch

from torch import nn
import numpy as np
import pandas as pd
import os

from tst.utils import *
import metric
import matplotlib.pyplot as plt
from data_processing.data_statices import NINAPRO_1
from dataloader import dataLoader, dataLoader_std, dataLoader_rms, dataLoader10_zjh, dataLoader23_zjh
import warnings
from matplotlib.font_manager import *
# myfont = FontProperties(fname='/home/ZJH/simsun.ttc')

def draw_graph(output, target):
    fig = plt.figure(figsize=(20, 10))
    for i in range(10):
        plt.subplot(5, 2, i + 1)
        # plt.plot(total_output_test[:, i].detach().cpu().numpy(), label="predict", color="b")
        # plt.plot(total_target_test[:, i].detach().cpu().numpy(), label="truth", color="r")
        plt.plot(output[:, i], label="predict", color="b")
        plt.plot(target[:, i], label="truth", color="r")
        # print(total_output_test[:, i].detach().cpu().numpy().shape)
    return fig

def mere_test(model='LSTM',obj = 1):
    # hidden_size = 64
    # num_layers = 1
    # d_feat = 12
    class_num = 10
    # dropout = 0.3
    # len_seq = 200
    # loss_type = 'mmd'
    #obj = 1
    batch_size = 32
    Time_Step = 100
    Windows = 200
    Model_Name = model  # 'Transform'
    model_str = ''
    if Model_Name == 'LSTM':
        #model_str = './hesi/LSTMsub' + str(obj) + '128_3_0.3.pkl'
        #model_str = './hesi/LSTMsub' + str(obj) + '_64_3_0.3.pkl'
        model_str = './save/new/LSTM/LSTMz-zero_sub' + str(obj) + '_32_3_0.3.pkl'
    if Model_Name == 'GRU':
        #model_str = './hesi/GRUsub' + str(obj) + '128_3_0.3.pkl'
        #model_str = './hesi/GRUsub' + str(obj) + '_64_3_0.3.pkl'
        model_str = './save/new/GRU/GRUz-zero_sub' + str(obj) + '_32_3_0.3.pkl'
    if Model_Name == 'LE-Conv_MN':
        model_str = './save/new/LE-Conv_MN/LE-ConvMNmiusub'+str(obj) + '_1_3_0.3.pkl'
    if Model_Name == 'LE-LSTM':
        #model_str = './hesi/LE-LSTM__128sub' + str(obj) + '128_3_0.3.pkl'
        #model_str = './hesi/LE-LSTM__sub' + str(obj) + '_64_3_0.3.pkl'
        # LE-LSTMmiu__sub8_1_3_0.3.pkl
        model_str = './save/new/LE-LSTM/LE-LSTMz-zero__sub' + str(obj) + '_1_3_0.3.pkl'
    if Model_Name == 'LE-GRU':
        #model_str = './hesi/LE-GRU__128sub' + str(obj) + '128_3_0.3.pkl'
        #model_str = './hesi/LE-GRU__sub' + str(obj) + '_64_3_0.3.pkl'
        model_str = './my_pkl/LE-GRUmiu__sub' + str(obj) + '_1_3_0.3.pkl'
    if Model_Name == 'Transformer_EMSA_UN':
        ## Transformer_EMSAUFNN_lays2_DWT_64_sub40_batch32z-zero.pkl
        model_str = './save/new/AblationStudy/Transformer_EMSA_lays2_MSE_128_sub' + str(obj) + '_batch32z-zero.pkl'
        # model_str = './save/Transform_chunk_sub_' + str(obj) + '_64_1_128_self.pkl'
    if Model_Name == 'Transformer_EMSA':
        # model_str = './my_pkl/ZJH_40/Transformer_EMSA_lays3_MSE_sub' + str(obj) + '_batch32miu.pkl'
        # model_str = './my_pkl/ZJH_40_Zero/Soft_dwt/Transformer_EMSA_lays3_DWT_sub' + str(obj) + '_batch32z-zero.pkl'
        # model_str = './my_pkl/ZJH_40_mix/Transformer_EMSA_lays3_MSE_sub' + str(obj) + '_batch32min-max.pkl'
        # model_str = './my_pkl/ZJH_40_Zero/MSE/Transformer_EMSA_lays3_MSE_sub' + str(obj) + '_batch128z-zero.pkl'
        # model_str = './save/new/myTransformer/Transformer_EMSA_lays3_MSE_sub'+str(obj)+'_batch32miu.pkl'
        model_str = './save/new/myTransformer/Transformer_EMSA_lays2_DWT_128_sub'+str(obj)+'_batch32z-zero.pkl'
        # model_str = './save/new/myTransformer/Transformer_EMSA_lays3_MSE_sub'+str(obj)+'_batch32miu.pkl'
        #print(model_str)
    if Model_Name == 'Transformer_window':
        #model_str = './save/new/myTransformer/TransformerWindow_lays2_MSE_128_sub' + str(obj) + '_batch32z-zero.pkl'
        model_str = './save/new/myTransformer/Transformer_Windows_lays3_MSE_sub' + str(obj) + '_batch32miu.pkl'
    if Model_Name == 'Transformer':
        # Stransformer_EMSA_sub_1_1_1_128.pkl
        # model_str = './hesi/Transformer_EMSA_sub_'+str(obj)+'_1_1_128.pkl'
        model_str = './save/new/myTransformer/Transformer_lays3_MSE_sub' + str(obj) + '_batch32z-zero.pkl'
    if Model_Name == 'Transformer_EMSA_dwt':
        model_str = './hesi/Transformer_EMSA_dwt_sub_'+str(obj)+'_2_1_128.pkl'
    if Model_Name == 'TCN':
        model_str = './save/new/TCN/TCN_miusub' + str(obj) + '_3_.pkl' #TCN_miusub39_3_.pkl
        # model_str = './save/new/TCN/TCN0306_z-zerosub' + str(obj) + '_3_.pkl'
    if Model_Name == 'LS-TCN':
        model_str = './save/new/LS-TCN/LS-TCN0306_z-zerosub'+str(obj) + '_31_.pkl'
    if Model_Name == 'BERT':
        model_str = './save/new/BERT/BERT_sub'+str(obj)+'_32z-zero_3_0.3.pkl'
        # model_str = './save/new/BERT_EMSA/BERT_sub'+str(obj)+'_32miu_3_0.3.pkl'

    model = torch.load(model_str).cuda()
    print(model)
    # data_read_test=ninapro_12_200.NINAPRO_1('data'+'/ninapro/S'+str(obj)+'emgtrain_rms.csv','data'+'/ninapro/S'+str(obj)+'glovetrain_rms.csv','data'+'/ninapro/S'+str(obj)+'emgtest_rms.csv','data'+'/ninapro/S'+str(obj)+'glovetest_rms.csv',Time_Step,Windows,train=False,Normalization=False)
    # test = DataLoader(data_read_test,batch_size=batch_size,drop_last=True)
    # print(model)
    train_data, test_data = dataLoader10_zjh(obj=obj, BATCH_SIZE=batch_size, TIME_STEP=Time_Step, WINDOW_SIZE=Windows,normal = "z-zero")

    total_output_test = torch.Tensor([]).cuda()
    total_target_test = torch.Tensor([]).cuda()
    cc_test = 0.0
    aver_test = 0.0
    t_hidden = None
    with torch.no_grad():
        for step, (data, target) in enumerate(test_data):
            x = data.type(torch.FloatTensor)
            y = target.permute(0, 1, 2).type(torch.FloatTensor)
            indices = torch.LongTensor([1, 2, 4, 5, 7, 8, 11, 12, 15, 16])
            y = torch.index_select(y[:, 199:200, :], 2, indices).cuda()        
            # output_test= model.predict(x.cuda())
            if Model_Name == 'Transformer':
                # print(x.shape)
                output_test, _ = model(x.cuda())
                # print(output_test)
            if Model_Name == 'LSTM':
                output_test = model(x.cuda())
            if Model_Name == 'TCN':
                x = x.permute(0, 2, 1).type(torch.FloatTensor)
                output_test = model(x.cuda())
            if Model_Name == 'LS-TCN':
                x = x.permute(0, 2, 1).type(torch.FloatTensor)
                output_test = model(x.cuda())
            if Model_Name == 'GRU':
                output_test = model(x.cuda())
            if Model_Name == 'LE-LSTM':
                output_test,t_hidden = model(x.cuda(),t_hidden)
            if Model_Name == 'LE-GRU':
                output_test,t_hidden = model(x.cuda(),t_hidden)
            if Model_Name == 'LE-Conv_MN':
                x = torch.unsqueeze(x.cuda(),3)
                output_test,t_hidden = model(x.cuda(),t_hidden)
            if Model_Name == 'Transformer_EMSA_UN':
                output_test,_ = model(x.cuda())
            if Model_Name == 'Transformer_window':
                output_test,_ = model(x.cuda())
            if Model_Name == 'Transformer_EMSA':
                output_test,_ = model(x.cuda())
            if Model_Name == 'STransformer':
                output_test,_ = model(x.cuda())
            if Model_Name == 'Transformer_EMSA_dwt':
                output_test,_ = model(x.cuda())
            if Model_Name == 'BERT':
                output_test,_ = model(x.cuda())
            total_output_test = torch.cat([total_output_test, output_test.view([batch_size, class_num])])
            total_target_test = torch.cat([total_target_test, y.view([batch_size, class_num]).cuda()])
        # for id in range(10):
        #     cc_test = metric.pearsonr(total_output_test[:, id], total_target_test[:, id])
        #     #print(cc_test)
        #     aver_test+=cc_test
        # print(aver_test/10)
    
    std = []
    means = []
    NINAPRO_1('./data/ninapro10_40' + '/S' + str(obj) + '_E2_A_glove_test.csv', std=std, mean=means)
    std_index = []
    means_index = []
    # indices = [1, 2, 4, 5, 7, 8, 11, 12, 15, 16]
    for index in range(10):
        print(indices[index])
        std_index.append(std[indices[index]])
        means_index.append(means[indices[index]])
    # print(std_index)
    # print(means_index)
    
    for index in range(10):
        total_target_test[:, index] = means_index[index] + std_index[index] * total_target_test[:, index]
        total_output_test[:, index] = means_index[index] + std_index[index] * total_output_test[:, index]
    # total_output_test = avg_smoothing_np(2, total_output_test) 
    cc_test = 0.0
    aver_test = 0.0
    aver_rmse = 0.0
    aver_nrmse = 0.0
    cc = []
    rmse_ls = []
    nrmse_ls = []
    total_output_test = avg_smoothing_np(3, total_output_test)
    for id in range(10):
        cc_test = metric.pearsonr(total_output_test[:, id], total_target_test[:, id])
        print("index:",id,format(cc_test.item(),'.4f'))

        rmse = metric.RMSE(total_output_test[:, id], total_target_test[:, id])

        nrmse = rmse / (torch.max(total_target_test[:, id]) - torch.min(total_target_test[:, id]))
        aver_test += cc_test
        aver_rmse += rmse
        aver_nrmse += nrmse
        cc.append(float(format(cc_test.item(), '.4f')))
        rmse_ls.append(float(format(rmse.item(), '.4f')))
        nrmse_ls.append(float(format(nrmse.item(), '.4f')))
    print(cc)
    print(rmse_ls)
    print(nrmse_ls)
    # fig = plt.figure(figsize=(20, 10))
    fig,ax = plt.subplots(1,10,figsize=(20,10))
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = ['SimSun'] # 修改字体为宋体
    # plt.rcParams['axes.unicode_minus'] = False	# 正常显示 '-'
    # plt.xticks(fontproperties=myfont)
    for i in range(4):
        ax[i] = plt.subplot(5, 2, i + 1)
        # plt.xlim(0,len(total_target_test[:, i].cpu().numpy())+500)
        # plt.ylim(-50,50)
        
        ax[i].plot(total_output_test[:, i].detach().cpu().numpy(), color = 'b')
        ax[i].plot(total_target_test[:, i].detach().cpu().numpy(), color = 'r')
        ax[i].set_xlabel(u'Sampling points')  # fontproperties=myfont,, fontsize=12
        ax[i].set_ylabel(u'Angle Amplitude(°)') # fontproperties=myfont, fontsize=12
        # ax[i].legend()
    # fig.set_xlabel(u'采样点', fontproperties=myfont,fontsize=12)
    # fig.set_ylabel(u'关节角度(°)', fontproperties=myfont,fontsize=12)
    # fig.savefig("./figure/new/TCN" + str(obj)+"Tes11")
    # fig = draw_graph(total_output_test.detach().cpu().numpy(), total_target_test.detach().cpu().numpy())
    # fig.savefig("./figure/LSTCN" + str(obj)+"Test")
    aver_test /= 10.0
    aver_rmse /= 10.0
    aver_nrmse /= 10.0
    print("OBJ", obj, "CC:", aver_test, "RMSE:", aver_rmse, "NRMSE:", aver_nrmse)
    return aver_test,aver_rmse,aver_nrmse
if __name__ == "__main__":
    # obj = [1, 5, 13, 14, 17, 19, 23, 33]
    #obj = [1, 5, 7, 9, 17, 19, 23, 24]
    # obj = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
    # obj = [1,2,3,4,5,6,7,8,9,10]
    obj = [5]
    # obj = [1,4,13,14,22,26,30,36]
    obj = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,33,34,35,36,37,38,39,40]
    # main_transfer(args,args.obj)
    # train(obj)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='LSTM')
    parser.add_argument('--gpu_id',type=int,default='3')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    model_name = args.model_name
    aver_cc = []
    aver_rmse = []
    aver_nrmse = []
    aver_cc_list = []
    aver_rmse_list = []
    aver_nrmse_list = []
    for i in range(len(obj)):
        cc,rmse,nrmse = mere_test(model=model_name,obj=obj[i])
        aver_cc.append(cc.cpu().numpy())
        aver_cc_list.append(float(format(cc.item(), '.4f')))
        aver_rmse.append(rmse.cpu().numpy())
        aver_rmse_list.append(float(format(rmse.item(), '.4f')))
        aver_nrmse.append(nrmse.cpu().numpy())
        aver_nrmse_list.append(float(format(nrmse.item(), '.4f')))
    for i in range(len(obj)):
        print("这是第",obj[i],"个人:","CC:",aver_cc[i],"RMSE:",aver_rmse[i],"NRMSE:",aver_nrmse[i])
    print("Model_Name:",model_name,"Average_CC:",np.sum(aver_cc)/len(obj),"Average_RMSE",np.sum(aver_rmse)/len(obj),"Average_NRMSE:",np.sum(aver_nrmse)/len(obj))
    print(aver_cc_list)
    print(aver_rmse_list)
    print(aver_nrmse_list)

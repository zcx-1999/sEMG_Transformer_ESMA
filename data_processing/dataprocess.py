import os

from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset.data_act as data_act
import pandas as pd
import dataset.data_weather as data_weather
import datetime
import numpy as np
from base.loss_transfer import TransferLoss
import torch
import math
import matplotlib.pyplot as plt
from data_processing import ninapro_12_200, ninapro_adarnn


def split_data(num_domain = 6,mode='tdc', data_train = None,data_test=None,dis_type = 'mmd'):
    if mode == 'tdc':
        return TDC(num_domain, data_train,data_test, dis_type = dis_type)
    else:
        print("error in mode")
def TDC(num_domain,train_data,test_data,dis_type='mmd'):
    start_time = 0  #0
    # emg,glove = data_file
    #end_time = data_file.numpy().shape[0]
    train_sum =  train_data.numpy().shape[0]
    #print(train_sum/77000)
    test_num = test_data.numpy().shape[0]
    # feat = torch.Tensor(data_file,dtype=torch.float32)
    split_N= 10
    selected = [0, 10]
    candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    start = 0
    if num_domain in [2, 3, 5, 6, 7, 10]:
        while len(selected) - 2 < num_domain - 1:
            distance_list = []
            for can in candidate:
                selected.append(can)
                selected.sort()
                dis_temp = 0
                for i in range(1, len(selected) - 1):
                    for j in range(i, len(selected) - 1):
                        index_part1_start = start + math.floor(selected[i - 1]/ split_N * train_sum)
                        index_part1_end = start + math.floor(selected[i]/ split_N * train_sum)
                        feat_part1 = train_data[index_part1_start: index_part1_end]
                        index_part2_start = start + math.floor(selected[j]/ split_N * train_sum)
                        index_part2_end = start + math.floor(selected[j + 1]/ split_N * train_sum)
                        feat_part2 = train_data[index_part2_start:index_part2_end]
                        # print(feat_part1.shape[1])
                        criterion_transder = TransferLoss(loss_type=dis_type, input_dim=feat_part1.shape[1])
                        dis_temp += criterion_transder.compute(torch.tensor(feat_part1, dtype=torch.float32),
                                                               torch.tensor(feat_part2, dtype=torch.float32))
                distance_list.append(dis_temp)
                selected.remove(can)
            can_index = distance_list.index(max(distance_list))
            # can_index = distance_list.index(min(distance_list))
            selected.append(candidate[can_index])
            candidate.remove(candidate[can_index])
        selected.sort()
        res_train = []
        res_test = []
        for i in range(1,len(selected)):
            if i == 1:
                sel_start_time = start_time + int(train_sum / split_N * selected[i - 1])
                test_start_time = start_time + (int(test_num / split_N * selected[i - 1]))
            else:
                sel_start_time = start_time + int(train_sum/ split_N * selected[i - 1])+1
                test_start_time = start_time + int(test_num/ split_N * selected[i - 1])+1
            sel_end_time = start_time + int(train_sum/ split_N * selected[i])
            test_end_time = start_time + int(test_num/ split_N * selected[i])
            # sel_start_time = datetime.datetime.strftime(sel_start_time,'%Y-%m-%d %H:%M')
            # sel_end_time = datetime.datetime.strftime(sel_end_time,'%Y-%m-%d %H:%M')
            res_train.append((sel_start_time, sel_end_time))
            res_test.append((test_start_time,test_end_time))
        #print(res)

        print(res_train)
        print(res_test)
        return res_train,res_test
    else:
        print("error in number of domain")
def get_split_emg_data(data,start,end,batch_size):
    ls_data = data[start:end:,]
    return ls_data
def get_split_glove_data(data,start,end,batch_size):
    ls = data[start:end:,]
    #print(ls)
    # fig = plt.figure(figsize=(20, 10))
    # for i in range(10):
    #     plt.subplot(5, 2, i + 1)
    #     plt.plot(ls[:, i].detach().cpu().numpy())
    #     plt.plot(ls[:, i].detach().cpu().numpy())
    # fig.savefig("result_AdaRNN_train_glove" )
    return ls

def p_test_loader(data,batch_size):
    test_loader = DataLoader(data,batch_size,shuffle=False,drop_last=True)
    return test_loader
def read_data(EMGtrain_dir, Glovetrain_dir, EMGtest_dir, Glovetest_dir):
    # pd.set_option('precision', 8)
    #print('data have not been processed')
    EMGtrain_data = torch.from_numpy(np.array(pd.read_csv(EMGtrain_dir, header=None)))
    #print(EMGtrain_dir + 'is loaded')
    Glovetrain_data = torch.from_numpy(np.array(pd.read_csv(Glovetrain_dir, header=None)))
    #print(Glovetrain_dir + 'is loaded')
    EMGtest_data = torch.from_numpy(np.array(pd.read_csv(EMGtest_dir, header=None)))
    #print(EMGtest_dir + 'is loaded')
    Glovetest_data = torch.from_numpy(np.array(pd.read_csv(Glovetest_dir, header=None)))
    #print(Glovetest_dir + 'is loaded')
    for index in range(12):
        EMGtrain_data[:, index] = (EMGtrain_data[:, index] - torch.min(EMGtrain_data[:, index])) / (
                    torch.max(EMGtrain_data[:, index]) - torch.min(EMGtrain_data[:, index]))
        EMGtest_data[:, index] = (EMGtest_data[:, index] - torch.min(EMGtest_data[:, index])) / (
                    torch.max(EMGtest_data[:, index]) - torch.min(EMGtest_data[:, index]))
    for index in range(22):
        # print(index)
        Glovetest_data[:, index] = (Glovetest_data[:, index] - torch.min(Glovetest_data[:, index])) / (
                    torch.max(Glovetest_data[:, index]) - torch.min(Glovetest_data[:, index]))
        Glovetrain_data[:, index] = (Glovetrain_data[:, index] - torch.min(
            Glovetrain_data[:, index])) / (torch.max(Glovetrain_data[:, index]) - torch.min(
            Glovetrain_data[:, index]))
    # self.EMGtrain_data=torch._cast_Float(self.EMGtrain_data)
    # print(EMGtrain_data)
    # print(Glovetrain_data)
    return EMGtrain_data,EMGtest_data,Glovetrain_data,Glovetest_data
def load_weather_data_multi_domain(obj = 1, batch_size=128,WINDOW_SIZE = 200,TIME_STEP = 100 , number_domain=6, mode='tdc', dis_type ='mmd'):
    # mode: 'auto', 'pre_process'
    #data_file = os.path.join(file_path, "PRSA_Data_1.pkl")
    #mean_train, std_train = data_weather.get_weather_data_statistic(data_file, station=station, start_time='2013-3-1 0:0',
    #                                                               end_time='2016-10-30 23:0')
    # EMGtrain_data,EMGtest_data,Glovetrain_data,Glovetest_data= read_data('../data/ninapro' + '/S' + str(obj) + 'emgtrain_rms.csv',
    #                                           '../data/ninapro' + '/S' + str(obj) + 'glovetrain_rms.csv',
    #                                           '../data/ninapro' + '/S' + str(obj) + 'emgtest_rms.csv',
    #                                           '../data/ninapro' + '/S' + str(obj) + 'glovetest_rms.csv')
    EMGtrain_data,EMGtest_data,Glovetrain_data,Glovetest_data= read_data('./data/ninapro' + '/S' + str(obj) + 'emgtrain_rms.csv',
                                              './data/ninapro' + '/S' + str(obj) + 'glovetrain_rms.csv',
                                              './data/ninapro' + '/S' + str(obj) + 'emgtest_rms.csv',
                                              './data/ninapro' + '/S' + str(obj) + 'glovetest_rms.csv')
    print(Glovetrain_data.numpy().shape)
    train_time,test_time = split_data(number_domain, mode=mode, data_train=Glovetrain_data,data_test=EMGtest_data, dis_type = dis_type)
    #print(train_time)
    #print(split_time_list)
    #print(len(split_time_list))
    #train_time = []
    train_list = []
    #test_list = []
    for i in range(len(train_time)):
        train_time_temp = train_time[i]
        #test_time_temp = test_time[i]
        #print(time_temp[0],time_temp[1])
        emg_train = get_split_emg_data(EMGtrain_data,train_time_temp[0],train_time_temp[1],batch_size)
        glove_train = get_split_glove_data(Glovetrain_data,train_time_temp[0],train_time_temp[1],batch_size)
        #emg_test = get_split_glove_data(EMGtest_data,test_time_temp[0],test_time_temp[1],batch_size)
        #glove_test = get_split_emg_data(Glovetest_data,test_time_temp[0],test_time_temp[1],batch_size)
        train_loader = ninapro_adarnn.NINAPRO_1(emg_train,glove_train,EMGtest_data,Glovetest_data,TIME_STEP,WINDOW_SIZE,train=True)
        #test_loader = ninapro_adarnn.NINAPRO_1(emg_train,glove_train,emg_test,glove_test,TIME_STEP,WINDOW_SIZE,train=False)
        train_loader = DataLoader(dataset=train_loader,batch_size=batch_size,shuffle=False)
        #test_loader = DataLoader(dataset=test_loader,batch_size=batch_size,shuffle=False)
        train_list.append(train_loader)
        #test_list.append(test_loader)
    test_loader = ninapro_adarnn.NINAPRO_1(EMGtrain_data, Glovetrain_data, EMGtest_data, Glovetest_data, TIME_STEP, WINDOW_SIZE, train=False)
    test_loader = p_test_loader(test_loader,batch_size)
    return train_list, test_loader
if __name__ == '__main__':
    train_list,test_list = load_weather_data_multi_domain(obj=1,batch_size=128,WINDOW_SIZE=200,TIME_STEP=100,number_domain=6)
    len_loader = np.inf
    for loader in train_list:
        if len(loader) < len_loader:
            len_loader = len(loader)
    print(len_loader)
    for data_all in (train_list):
        for i,(fea,label) in enumerate(data_all):
            print(fea)
            print(fea.shape)
            print(label)
            print(label.shape)
    for data_all in tqdm(zip(*train_list), total=len_loader):
        for (emg,glove) in data_all:
            print(emg)
            print(emg.shape)
            print(glove)
            print(glove.shape)
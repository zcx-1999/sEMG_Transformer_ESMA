from __future__ import print_function
import torch.utils.data as data
import torch
import os
import pandas
import scipy.signal as signal
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
class NINAPRO_1(data.Dataset):

    def __init__(self, EMGtrain_dir, Glovetrain_dir, EMGtest_dir, Glovetest_dir, move, windowsize, Normalization=False, train=True):
        self.train = train
        # self.EMGtrain_dir = EMGtrain_dir
        # self.Glovetrain_dir = Glovetrain_dir
        # self.EMGtest_dir = EMGtest_dir
        # self.Glovetest_dir = Glovetest_dir
        self.move = move
        self.windowsize = windowsize
        # pd.set_option('precision', 8)
        #print('data have not been processed')
        self.EMGtrain_data = EMGtrain_dir
        #print('data is loaded')
        self.Glovetrain_data = Glovetrain_dir
        #print('data is loaded')
        self.EMGtest_data = EMGtest_dir
        #print('data is loaded')
        self.Glovetest_data = Glovetest_dir
        #print('data is loaded')
        #MaxMinNormalization
        if Normalization :
            # self.EMGtrain_data=torch._cast_Float((self.EMGtrain_data-torch.min(self.EMGtrain_data))/(torch.max(self.EMGtrain_data)-torch.min(self.EMGtrain_data)))
            # self.Glovetrain_data =torch._cast_Float((self.Glovetrain_data - torch.min(self.Glovetrain_data)) / (torch.max(self.Glovetrain_data) - torch.min(self.Glovetrain_data)))
            # self.EMGtest_data = torch._cast_Float((self.EMGtest_data - torch.min(self.EMGtest_data)) / (torch.max(self.EMGtest_data) - torch.min(self.EMGtest_data)))
            # self.Glovetest_data = torch._cast_Float((self.Glovetest_data - torch.min(self.Glovetest_data)) / (torch.max(self.Glovetest_data) - torch.min(self.Glovetest_data)))

            # self.EMGtrain_data = torch._cast_Float((self.EMGtrain_data - torch.mean(self.EMGtrain_data)) /
            #             #             torch.std(self.EMGtrain_data) )
            #             # self.Glovetrain_data = torch._cast_Float((self.Glovetrain_data - torch.mean(self.Glovetrain_data)) /
            #             #             torch.std(self.Glovetrain_data))
            #             # self.EMGtest_data = torch._cast_Float((self.EMGtest_data - torch.mean(self.EMGtest_data)) /
            #             #             torch.std(self.EMGtest_data))
            #             # self.Glovetest_data = torch._cast_Float((self.Glovetest_data - torch.mean(self.Glovetest_data)) /
            #             #             torch.std(self.Glovetest_data))

            # ## Z-Zero正则化
            # for index in range(12):
            #     #print(self.EMGtrain_data[:,index].shape)
            #
            #     self.EMGtrain_data[:,index] = self.Z_Score(self.EMGtrain_data[:,index])
            #     self.EMGtest_data[:,index] = self.Z_Score(self.EMGtest_data[:,index])
            # for index in range(22):
            #     self.Glovetest_data[:,index] = self.Z_Score(self.Glovetest_data[:,index])
            #     self.Glovetrain_data[:,index] = self.Z_Score(self.Glovetrain_data[:,index])
            # ### 最大最小正则化
            for index in range(12):
                self.EMGtrain_data[:,index]=(self.EMGtrain_data[:,index]-torch.min(self.EMGtrain_data[:,index]))/(torch.max(self.EMGtrain_data[:,index])-torch.min(self.EMGtrain_data[:,index]))
                self.EMGtest_data[:,index] = (self.EMGtest_data[:,index] - torch.min(self.EMGtest_data[:,index])) / (torch.max(self.EMGtest_data[:,index]) - torch.min(self.EMGtest_data[:,index]))
            for index in range(22):
                #print(index)
                self.Glovetest_data[:,index] =(self.Glovetest_data[:,index] - torch.min(self.Glovetest_data[:,index])) / (torch.max(self.Glovetest_data[:,index]) - torch.min(self.Glovetest_data[:,index]))
                self.Glovetrain_data[:,index] = (self.Glovetrain_data[:,index] - torch.min(self.Glovetrain_data[:,index])) / (torch.max(self.Glovetrain_data[:,index]) - torch.min(self.Glovetrain_data[:,index]))
            # self.EMGtrain_data=torch._cast_Float(self.EMGtrain_data)
            # self.EMGtest_data=torch._cast_Float(self.EMGtest_data)
            # self.Glovetrain_data=torch._cast_Float(self.Glovetrain_data)
            # self.Glovetest_data=torch._cast_Float(self.Glovetest_data)
    # self.train_data = []
        # for step in range(len(self.EMGtrain_data)):
        #     self.train_data.append(np.array(self.EMGtrain_data[step * move:(step) * move + windowsize]))
        # traindata=self.train_data
        #
        #
        # self.train_target=[]
        # for step in range(len(self.Glovetrain_data)):
        #     self.train_target.append(np.array(self.Glovetrain_data[step * move:(step) * move + windowsize]))
        # traintarget =self.train_target
        #
        # self.test_data=[]
        # for step in range(len(self.EMGtest_data)/windowsize):
        #     self.test_data.append(np.array(self.EMGtest_data[step * move:(step) * move + windowsize]))
        # #self.test_data=torch.Tensor(self.test_data)
        #
        # self.test_target=[]
        # for step in range(len(self.Glovetest_data)/windowsize):
        #     self.test_target.append(np.array(self.Glovetest_data[step * move:(step) * move + windowsize]))
        # #self.test_target=torch.Tensor(self.test_target)
        # print()


    def __getitem__(self, index):
        if self.train:
            emg, ang = self.EMGtrain_data[index*self.move:index*self.move+self.windowsize], self.Glovetrain_data[index*self.move:index*self.move+self.windowsize]
        else:
            emg, ang = self.EMGtest_data[index*self.move:index*self.move+self.windowsize], self.Glovetest_data[index*self.move:index*self.move+self.windowsize]
        return emg, ang

    def __len__(self):
        if self.train:
            return (len(self.EMGtrain_data)-self.windowsize)//self.move
        else:
            return (len(self.EMGtest_data)-self.windowsize)//self.move

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        # fmt_str += '    Root Location: {}\n'.format(self.root)
        # tmp = '    Transforms (if any): '
        # fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        # tmp = '    Target Transforms (if any): '
        # fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def Z_Score(self,data):
        lenth = data.shape[0]
        total = sum(data)
        ave = float(total) / lenth
        tempsum = sum([pow(data[i] - ave, 2) for i in range(lenth)])
        tempsum = pow(float(tempsum) / lenth, 0.5)
        for i in range(lenth):
            data[i] = (data[i] - ave) / tempsum
        return data

    def MaxMinNormalization(self, x):
        #min = x.min()
        #x = x / np.sqrt(np.sum((x-min)**2))+min
        # x = 2*((x - np.min(x)) / (np.max(x) - np.min(x)))-1
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return x

    def RootMeanSqure(self, data):
        ##print("sss")
        returnrms = 0
        for i in data:
            returnrms = returnrms + i ** 2
        returnrms = returnrms / len(data)
        returnrms = returnrms ** 0.5
        return returnrms

    def idd_reshape_to_origion(self, data, windowsize, overlap):
        batchnum = len(data)
        channelnum = len(data[1][1])
        # data = data.reshape([batchnum,windowsize,channelnum])
        returndata = []
        for n in range(channelnum):
            for i in range(batchnum):
                if i == 0:
                    tmpchanneldata = data[i, :, n]
                else:
                    tmpchanneldata = np.hstack((tmpchanneldata, data[i, overlap:, n]))
            returndata.append(tmpchanneldata)
        return np.squeeze(returndata)

    def view_my_data(self, data_x, data_y, windowsize, overlap):
        plt.figure(int(np.random.rand()*1000))
        xx = self.idd_reshape_to_origion(data_x, windowsize, overlap)
        yy = self.idd_reshape_to_origion(data_y, windowsize, overlap)
        for i in np.arange(len(data_x[0][0])):
            plt.plot(xx[i].squeeze()+i+1)
        for o in np.arange(len(data_y[0][0])):
            plt.plot(yy[o].squeeze()-o-1)

        plt.show()
        plt.savefig('the input.png')

    def savitzky_golay(self, y, window_size, order, deriv=0, rate=1):

        import numpy as np
        from math import factorial

        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError as msg:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order + 1)
        half_window = (window_size - 1) // 2
        # precompute coefficients
        b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
        m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve(m[::-1], y, mode='valid')

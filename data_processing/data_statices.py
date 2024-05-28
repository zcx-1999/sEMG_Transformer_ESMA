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

from tsmoothie.utils_func import sim_randomwalk
from tsmoothie.smoother import LowessSmoother


class NINAPRO_1(data.Dataset):

    def __init__(self, Glovetest_dir,std,mean):
        self.Glovetest_dir = Glovetest_dir
        # pd.set_option('precision', 8)
        self.std = std
        self.mean = mean
        self.Glovetest_data = torch.from_numpy(np.array(pd.read_csv(self.Glovetest_dir, header=None)))
        print(self.Glovetest_dir + 'is loaded')
        for index in range(22):
            std.append(torch.std(self.Glovetest_data[:,index]))
        for index in range(22):
            mean.append(torch.mean(self.Glovetest_data[:,index]))


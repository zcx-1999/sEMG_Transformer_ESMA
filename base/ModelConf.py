import os, sys

class ModelConf(object):

    """All LSTM/GRU + SelfAttention params"""

    def __init__(self):
        self.embedding_dim = 12  # 输入向量维度
        self.rnn_type = "gru"  # lstm, gru, no
        self.rnn_hidden = 128 # rnn 隐层维度
        self.attention_hidden = 128  # attention 隐层维度 # 256
        self.attention_num = 200  # attention 观点数 # 32
        self.penalty_C = 0.0  # attention矩阵二范数惩罚系数 # 0.0
        self.fc1_dim = 128  # 全连接维度(第1层) # 1024
        self.output_dim = 10  # 模块encode输出向量维度 # 10
        self.relu_leakiness = 0.0  # leaky_relu->negative_slope # 0.1
        self.use_dropout = False  # 是否使用dropout
        self.dropout_keep_prob = 0.7  # dropout保留概率

    def __str__(self):
        output = "embedding_dim: " + str(self.embedding_dim) + "\n" + \
                 "rnn_type: " + str(self.rnn_type) + "\n" + \
                 "rnn_hidden: " + str(self.rnn_hidden) + "\n" + \
                 "attention_hidden: " + str(self.attention_hidden) + "\n" + \
                 "attention_num: " + str(self.attention_num) + "\n" + \
                 "penalty_C: " + str(self.penalty_C) + "\n" + \
                 "fc1_dim: " + str(self.fc1_dim) + "\n" + \
                 "output_dim: " + str(self.output_dim) + "\n" + \
                 "relu_leakiness: " + str(self.relu_leakiness) + "\n" + \
                 "use_dropout: " + str(self.use_dropout) + "\n" + \
                 "dropout_keep_prob: " + str(self.dropout_keep_prob) + "\n"
        return output
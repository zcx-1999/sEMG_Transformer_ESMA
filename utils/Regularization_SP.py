import torch
import torch.nn as nn

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cuda'
print("-----device:{}".format(device))
print("-----Pytorch version:{}".format(torch.__version__))


class Regularization(torch.nn.Module):
    def __init__(self, srcmodel, tarmodel, weight_decay, p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.srcmodel = srcmodel
        self.tarmodel = tarmodel
        self.weight_decay = weight_decay
        self.p = p
        self.srcweight_list = self.get_newlayer_weight(srcmodel)
        self.tarweight_list = self.get_newlayer_weight(tarmodel)
        self.weight_info(self.srcweight_list)

    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, srcmodel, tarmodel):
        self.srcweight_list = self.get_newlayer_weight(srcmodel)  # 获得源域模型最新的权重
        self.tarweight_list = self.get_newlayer_weight(tarmodel)  # 获得目标域模型中最新的权重
        oldsrcweight = self.get_old_layer_weight(srcmodel)
        oldtarweight = self.get_old_layer_weight(tarmodel)

        reg_loss1 = self.regularization_loss(self.srcweight_list, self.tarweight_list, self.weight_decay, p=self.p)
        reg_loss2 = self.regularization_loss(oldsrcweight,oldtarweight,self.weight_decay,p = self.p)
        return reg_loss1 + reg_loss2

    def get_newlayer_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if param.requires_grad == False:
                continue
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
    def get_old_layer_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                continue
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, srcweight_list, tarweight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        le = len(srcweight_list)
        for i in range(le):

            for name, w in srcweight_list:
                src_w = w
            for name, w in tarweight_list:
                tar_w = w

            l2_reg = torch.norm(tar_w - src_w, p=p)

            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")
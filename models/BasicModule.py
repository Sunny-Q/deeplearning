#coding=utf8
import time
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        print(path)
        data = torch.load(path)
        self.load_state_dict(data)
        #return self.cuda() #when use cuda
        return self #don't use cuda

    def save(self, name=None):
        #prefix = 'snapshot/' + self.model_name + '_' +self.opt.type_+'_'
        prefix='textcnn'
        if name is None:
            name = time.strftime('%m%d_%H_%M_%S.pth')

        path = os.path.join(prefix,name)
        data=self.state_dict()

        torch.save(data, path)
        return path

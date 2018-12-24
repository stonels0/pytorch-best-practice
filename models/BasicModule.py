# -*- coding: utf-8 -*-

import torch as t
import time


class BasicModule(t.nn.Module):
    '''
    简单封装了nn.Module,主要提供了save和load两个方法
    '''

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))  # 默认名字

    def load(self, path):
        '''
        可加载指定路径下的模型
        '''
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        '''
        保存模型，默认使用"模型名称+时间"作为文件名
        如AlexNet_0709_23:59:29.pth
        '''
        if name is None:
            prefix = './checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m$d_%H:$M:%S.pth')

        # 建议保存对应的state_dict,而非直接保存整个 Module/Optimizer对象
        t.save(self.state_dict(), name)
        return name


class Flat(t.nn.Module):
    '''
    把输入 reshape成（batch_size, dim_length)
    '''

    def __init__(self):
        super(Flat, self).__init__()
        # self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)
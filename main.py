# -*- coding: utf-8 -*-
from __future__ import print_function
from config import opt
import os
import torch as t
import models
from data.dataset import DogCat
from torch.utils.data import Dataloader
from torchnet import meter
from utils.visualize import Visualize
from tqdm import tqdm

import torch

# at beginning of the script 
# device = torch.device('cuda:0" if torch.cuda.is_available() else "cpu")
current_GPU = torch.cuda.current_device()
# device = torch.device("cuda:{}".format(current_GPU) if torch.cuda.is_available() else "cpu")


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)


def train(**kwargs):
    # 根据命令行参数更新配置
    opt.parse(kwargs)
    vis = Visualize(opt.env)

    # step1: configure model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)

    device = torch.device("cuda:{}".format(current_GPU) if torch.cuda.is_available() and opt.use_gpu else "cpu")
    model.to(device)

    # step2: data
    # 通过 Dataloader加载数据，train参数控制， 训练|验证
    train_data = DogCat(opt.train_data_root, train=True)
    val_data = DogCat(opt.train_data_root, train=False)

    train_dataloader = Dataloader(
        train_data,
        opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers
    )

    val_dataloader = Dataloader(
        val_data,
        opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
    )

    # step3: criterion and optimizer
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(
                            model.parameters(), 
                            lr=lr,
                            weight_decay=opt.weight_decay
                            )

    # step4: meters 统计指标： 平滑处理之后的损失，还有混淆矩阵
    # 工具：meter，提供了一些轻量级的工具，用于帮助用户快速统计训练过程中的一些指标
    # meter.AverageValueMeter(均值、标准差计算)：能够计算所有数的平均值和标准差，这里用来统计一个 epoch中损失的平均值
    # meter.ConfusionMeter（混淆矩阵）:用来统计分类问题中的分类情况，是一个比 准确率更详细的 统计指标 (可理解为：真值图)
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # train
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in tqdm(enumerate(train_dataloader), total=len(train_data)):

            # train
            input = data.to(device)
            target = label.to(device)

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            # meters update and visualize
            # loss_meter.aded(loss.data[0])
            loss_meter.add(loss.item())    # 记录每个batch_size的损失 以计算平均损失
            # confusion_matrix.add(score.data, target.data)
            confusion_matrix.add(score.detach(), target.detach())   # 添加（模型输出，实际标注）

            # 每 print_frep 个batch_size 打印一次损失（可视化）
            if ii % opt.print_frep == opt.print_frep-1:
                vis.plot('loss', loss_meter.value()[0])

                # 进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()

        model.save()

        # validate and visualize
        val_cm, val_accuracy = val(model, val_dataloader)

        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch}, lr:{lr}, loss:{loss}, train_cm:{train_cm}, val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()), lr=lr
        ))

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法： 不会有 moment等信息的损失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]

    def val(model, dataloader):
        '''
        计算模型在验证集上的准确率等信息\n
        返回值：（混淆矩阵，准确率）
        '''
        # 将模型设为验证模式
        model.eval()
        confusion_matrix = meter.ConfusionMeter(2)

        for ii, data in enumerate(dataloader):
            input, lable = data

            # val_input = Variable(input, volatile=True)
            with t.no_grad():
                val_input = input
                val_label = label.type(t.LongTensor)

            if opt.use_gpu:
                val_input = val_input.cuda()
                val_label = val_label.cuda()

            score = model(val_input)
            # confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))
            confusion_matrix.add(\
                score.detach().squeeze(),\
                lable.type(t.LongTensor))
        # 验证完成后，将模型设置为训练模式
        # 因为会影响 BatchNorm 和 Dropout等层的运行模式
        model.train()
        cm_value = confusion_matrix.value()
        accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())

        return confusion_matrix, accuracy

    def test(**kwargs):
    	'''
        测试时，计算每个样本属于狗的概率，将结果保存成csv文件\n
        同验证代码大致相同，但是需要：加载模型和数据
        '''
        opt.parse(kwargs)

        import ipdb
        ipdb.set_trace()

        # configure model
        model = getattr(models, opt.model)().eval()    # 将模型设为 验证模式

        if opt.load_model_path:
            model.load(opt.load_model_path)

        device = torch.device("cuda:{}".format(current_GPU) if torch.cuda.is_available() and opt.use_gpu else "cpu")
        model.to(device)

        # data
        test_data = DogCat(opt.test_data_root, train=False, test=True)
        test_dataloader = Dataloader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers
        )
        results = []
        for ii, (data, path) in enumerate(test_dataloader):
            with t.no_grad:
                input = data.to(device)

            score = model(input)
            probability = t.nn.functional.softmax(score, dim=1)[:, 0].tolist()

            batch_results = [(path_, probability_) for path_, probability_ in zip(path, probability)]

            results += batch_results
        write_csv(results, opt.result_file)

        return len(results)

    def help():
        '''
        打印帮助的信息： python file.py help
        '''

        print('''
        usage: python file.py <function> [--args=value]
        <function> := train | test | help
        example:
                python {0} train --env='env1224' --lr=0.01
                python {0} test --dataset='path/to/dataset/root/'
                python {0} help
        avaiable args:'''.format(__file__))

        from inspect import getsource
        source = (getsource(opt.__class__))
        print(source)

    if __name__ == '__main__':
        import fire
        fire.Fire()
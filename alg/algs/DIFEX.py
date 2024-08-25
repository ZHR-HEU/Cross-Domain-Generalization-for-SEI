# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm
from alg.algs.DANN import DANN

class DIFEX(Algorithm):
    def __init__(self, args):
        super(DIFEX, self).__init__(args)
        self.args = args
        self.featurizer = get_fea(args)  #特征提取器
        #瓶颈层
        # self.bottleneck = common_network.feat_bottleneck(
        #     self.featurizer.in_features, args.bottleneck, args.layer)
        self.bottleneck = common_network.feat_bottleneck(300*8, 200, args.layer)
        # self.classifier = common_network.feat_classifier(
        #     args.num_classes, args.bottleneck, args.classifier)
        self.classifier = common_network.feat_classifier(
                 args.num_classes, 200, args.classifier)
        #特征一分为二
        # self.tfbd = args.bottleneck//2
        self.tfbd = 100
        #教师网络
        self.teaf = get_fea(args)
        # self.teab = common_network.feat_bottleneck(
        #     self.featurizer.in_features, self.tfbd, args.layer)
        self.teab = common_network.feat_bottleneck(300*8, self.tfbd, args.layer)
        self.teac = common_network.feat_classifier(
            args.num_classes, self.tfbd, args.classifier)
        self.teaNet = nn.Sequential(
            self.teaf,
            self.teab,
            self.teac
        )

    #教师网络训练
    def teanettrain(self, dataloaders, epochs, opt1, sch1):
        self.teaNet.train()
        minibatches_iterator = zip(*dataloaders)  #从多个数据集中加载数据
        for epoch in range(epochs):
            minibatches = [(tdata) for tdata in next(minibatches_iterator)]
            # minibatches = dataloaders
            all_x = torch.cat([data[0].cuda().float() for data in minibatches])  #合并输入数据
            # all_z = torch.angle(torch.fft.fftn(all_x, dim=(2, 3)))  # 傅里叶变换
            all_z = torch.angle(torch.fft.fftn(all_x, dim=(1, 2)))    #傅里叶变换
            all_y = torch.cat([data[1].cuda().long() for data in minibatches])  #合并输入标签
            all_p = self.teaNet(all_z)   #输入到教师网络中预测

            # all_p = torch.argmax(all_p, dim=1)
            # all_p = torch.tensor(all_p.float(), requires_grad=True)
            # all_p = all_p.float()
            all_y = all_y.squeeze()

            loss_class = torch.nn.NLLLoss()
            loss_class = loss_class.cuda()
            loss = loss_class(all_p, all_y)
            # loss = F.cross_entropy(all_p, all_y, reduction='mean')  #使用预测结果 all_p 和真实标签 all_y 来计算交叉熵损失。
            opt1.zero_grad()
            loss.backward()
            if ((epoch+1) % (int(self.args.steps_per_epoch*self.args.max_epoch*0.7)) == 0 or (epoch+1) % (int(self.args.steps_per_epoch*self.args.max_epoch*0.9)) == 0) and (not self.args.schuse):
                for param_group in opt1.param_groups:
                    param_group['lr'] = param_group['lr']*0.1
            opt1.step()
            if sch1:
                sch1.step()

            if epoch % int(self.args.steps_per_epoch) == 0 or epoch == epochs-1:
                print('epoch: %d, cls loss: %.4f' % (epoch, loss))
        self.teaNet.eval()

    #特征对齐
    def coral(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        Kxx = self.gaussian_kernel(x, x).mean()
        Kyy = self.gaussian_kernel(y, y).mean()
        Kxy = self.gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])  #合并输入的数据
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])   #合并输入的标签
        all_y = all_y.squeeze()
        #使用傅里叶变换处理输入数据，然后通过两个神经网络层（teaf 和 teab）进行特征提取---教师网络
        with torch.no_grad():
            all_x1 = torch.angle(torch.fft.fftn(all_x, dim=(1, 2)))
            tfea = self.teab(self.teaf(all_x1)).detach()

        all_z = self.bottleneck(self.featurizer(all_x))          #特征提取--普通网络
        all_z1 = self.classifier(all_z)
        # all_z1 = torch.argmax(all_z1, dim=1)
        # all_z1 = torch.tensor(all_z1.float(), requires_grad=True)

        loss_class = torch.nn.NLLLoss()
        loss_class = loss_class.cuda()
        loss1 = loss_class(all_z1, all_y)

        # loss1 = F.cross_entropy(all_z1, all_y)   #分类损失
        loss2 = F.mse_loss(all_z[:, :self.tfbd], tfea)*self.args.alpha  #教师和学生之间的MSE损失
        #正则化项，两部分特征差别尽量大
        if self.args.disttype == '2-norm':
            loss3 = -F.mse_loss(all_z[:, :self.tfbd],
                                all_z[:, self.tfbd:])*self.args.beta
        elif self.args.disttype == 'norm-2-norm':
            loss3 = -F.mse_loss(all_z[:, :self.tfbd]/torch.norm(all_z[:, :self.tfbd], dim=1, keepdim=True),
                                all_z[:, self.tfbd:]/torch.norm(all_z[:, self.tfbd:], dim=1, keepdim=True))*self.args.beta
        elif self.args.disttype == 'norm-1-norm':
            loss3 = -F.l1_loss(all_z[:, :self.tfbd]/torch.norm(all_z[:, :self.tfbd], dim=1, keepdim=True),
                               all_z[:, self.tfbd:]/torch.norm(all_z[:, self.tfbd:], dim=1, keepdim=True))*self.args.beta
        elif self.args.disttype == 'cos':
            loss3 = torch.mean(F.cosine_similarity(
                all_z[:, :self.tfbd], all_z[:, self.tfbd:]))*self.args.beta
        #对齐损失
        # loss4 = 0
        # if len(minibatches) > 1:
        #     for i in range(len(minibatches)-1):
        #         for j in range(i+1, len(minibatches)):
        #             loss4 += self.coral(all_z[i*self.args.batch_size:(i+1)*self.args.batch_size, self.tfbd:],
        #                                 all_z[j*self.args.batch_size:(j+1)*self.args.batch_size, self.tfbd:])
        #     loss4 = loss4*2/(len(minibatches) *
        #                      (len(minibatches)-1))*self.args.lam
        # else:
        #     loss4 = self.coral(all_z[:self.args.batch_size//2, self.tfbd:],
        #                        all_z[self.args.batch_size//2:, self.tfbd:])
        #     loss4 = loss4*self.args.lam

        loss4 = 0
        if len(minibatches) > 1:
            for i in range(len(minibatches)-1):
                for j in range(i+1, len(minibatches)):
                    loss4 += self.mmd(all_z[i*self.args.batch_size:(i+1)*self.args.batch_size, self.tfbd:],
                                        all_z[j*self.args.batch_size:(j+1)*self.args.batch_size, self.tfbd:])
            loss4 = loss4*2/(len(minibatches) *
                             (len(minibatches)-1))*self.args.lam
        else:
            loss4 = self.mmd(all_z[:self.args.batch_size//2, self.tfbd:],
                               all_z[self.args.batch_size//2:, self.tfbd:])
            loss4 = loss4*self.args.lam

        loss = loss1+loss2+loss3+loss4
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'class': loss1.item(), 'dist': (loss2).item(), 'exp': (loss3).item(), 'align': loss4.item(), 'total': loss.item()}

    def predict(self, x):
        return self.classifier(self.bottleneck(self.featurizer(x)))

# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, args):
        super(ERM, self).__init__(args)
        self.featurizer = get_fea(args)
        self.classifier = common_network.feat_classifier_two(
            args.num_classes, 100, args.classifier)

        self.network = nn.Sequential(
            self.featurizer, self.classifier)

    def update(self, minibatches, opt, sch):
        # for data in minibatches:
        #     data = data
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        all_y = all_y.squeeze()
        # model = self.network
        # for param in model.parameters():
        #     print(param.requires_grad)
        one_hot_targets = self.predict(all_x)
        # class_indices = torch.argmax(one_hot_targets, dim=1)
        class_indices = torch.tensor(one_hot_targets.float())

        loss_class = torch.nn.NLLLoss()
        loss_class = loss_class.cuda()
        loss = loss_class(one_hot_targets, all_y)
        #loss = F.cross_entropy(class_indices, all_y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        lossww = loss.item()
        return {'class': loss.item()}

    def predict(self, x):
        return self.network(x)

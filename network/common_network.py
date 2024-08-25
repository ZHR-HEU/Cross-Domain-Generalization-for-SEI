# coding=utf-8
import torch.nn as nn
from network.util import init_weights
import torch.nn.utils.weight_norm as weightNorm


class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim, type="ori"):
        super(feat_bottleneck, self).__init__()
        # self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        # self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(p=0.5)
        # self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        # self.bottleneck.apply(init_weights)
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(feature_dim, 300))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(300))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(300, bottleneck_dim))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(bottleneck_dim))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        # self.type = type


    def forward(self, x):
        # x = self.bottleneck(x)
        x = self.class_classifier(x)
        # if self.type == "bn":
        #     x = self.bn(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(
                nn.Linear(bottleneck_dim, class_num), name="weight")
            # self.fc.apply(init_weights)
        else:
            # self.fc = nn.Linear(bottleneck_dim, class_num)
            # self.fc.apply(init_weights)
            self.class_classifier = nn.Sequential()
            # self.class_classifier.add_module('c_fc1', nn.Linear(300 * 18, 300))
            # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(300))
            # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
            # self.class_classifier.add_module('c_drop1', nn.Dropout2d())
            # self.class_classifier.add_module('c_fc2', nn.Linear(300, 100))
            # self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
            # self.class_classifier.add_module('c_relu2', nn.ReLU(True))
            self.class_classifier.add_module('c_fc1', nn.Linear(bottleneck_dim, 7))
            self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

    def forward(self, x):
        x = self.class_classifier(x)
        # x = self.fc(x)
        return x


class feat_classifier_two(nn.Module):
    def __init__(self, class_num, input_dim, bottleneck_dim):
        super(feat_classifier_two, self).__init__()
        # self.type = type
        # self.fc0 = nn.Linear(input_dim, bottleneck_dim)
        # # self.fc0.apply(init_weights)
        # self.fc1 = nn.Linear(bottleneck_dim, class_num)
        # self.fc1.apply(init_weights)
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(300 * 8, 300))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(300))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(300, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 7))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

    def forward(self, x):
        # x = self.fc0(x)
        # x = self.fc1(x)
        x = self.class_classifier(x)
        return x

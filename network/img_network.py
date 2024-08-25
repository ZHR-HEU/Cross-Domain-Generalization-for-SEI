# coding=utf-8
import torch.nn as nn
from torchvision import models

vgg_dict = {"vgg11": models.vgg11, "vgg13": models.vgg13, "vgg16": models.vgg16, "vgg19": models.vgg19,
            "vgg11bn": models.vgg11_bn, "vgg13bn": models.vgg13_bn, "vgg16bn": models.vgg16_bn, "vgg19bn": models.vgg19_bn}


class VGGBase(nn.Module):
    def __init__(self, vgg_name):
        super(VGGBase, self).__init__()
        model_vgg = vgg_dict[vgg_name](pretrained=True)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), model_vgg.classifier[i])
        self.in_features = model_vgg.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


res_dict = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50,
            "resnet101": models.resnet101, "resnet152": models.resnet152, "resnext50": models.resnext50_32x4d, "resnext101": models.resnext101_32x8d}


class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class DTNBase(nn.Module):
    def __init__(self):
        super(DTNBase, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )
        self.in_features = 256*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x


class LeNetBase(nn.Module):
    def __init__(self):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.in_features = 50*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(2, 2 * 25, kernel_size=15, stride=1, padding=7, bias=True),
            nn.BatchNorm1d(2 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(2 * 25, 3 * 25, kernel_size=11, stride=1, padding=5, bias=True),
            nn.BatchNorm1d(3 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(3 * 25, 4 * 25, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm1d(4 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(4 * 25, 6 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(6 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            # nn.Dropout(0.25),

            nn.Conv1d(6 * 25, 8 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(8 * 25),
            nn.ReLU(inplace=True),
            nn.Conv1d(8 * 25, 8 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(8 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            # nn.Dropout(0.25),

            nn.Conv1d(8 * 25, 12 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(12 * 25),
            nn.ReLU(inplace=True),
            nn.Conv1d(12 * 25, 12 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(12 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(12 * 25, 12 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(12 * 25),
            nn.ReLU(inplace=True),
            nn.Conv1d(12 * 25, 12 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(12 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(12 * 25, 12 * 25, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm1d(12 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
        )
        # self.class_classifier = nn.Sequential()
        # self.class_classifier.add_module('c_fc1', nn.Linear(300 * 18, 300))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(300))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        # self.class_classifier.add_module('c_fc2', nn.Linear(300, 100))
        # self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        # self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        # self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        # self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

    def forward(self, x):
        x = x.expand(x.data.shape[0], 2, 2048)
        x = self.feature(x)
        # print(x.shape)
        # x = x.view(x.size(0), -1)
        x = x.view(-1, 300 * 8)
        # x = self.class_classifier(x)
        # print(x.shape)
        return x



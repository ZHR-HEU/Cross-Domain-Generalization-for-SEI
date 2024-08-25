# coding=utf-8
import torch
from network import img_network


# def get_fea(args):
#     if args.dataset == 'dg5':
#         net = img_network.DTNBase()
#     elif args.net.startswith('res'):
#         net = img_network.ResBase(args.net)
#     else:
#         net = img_network.VGGBase(args.net)
#     return net
def get_fea(args):
    net = img_network.CNNModel()
    return net


def accuracy(network, loader):
    correct = 0
    total = 0
    total1 = 0
    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            p = network.predict(x)
            y = y.squeeze()
            # correct1 =  (p.argmax(1).eq(y).float()).sum().item()
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
            total1 += 1

    network.train()
    return correct / total

# def accuracy(network, loader):
#     correct = 0
#     total = 0
#     total1, correct1 = 0, 0
#     network.eval()
#     with torch.no_grad():
#         # x = torch.cat([data[0].cuda().float() for data in loader])
#         # y = torch.cat([data[1].cuda().long() for data in loader])
#         x = loader[0].cuda().float()
#         y = loader[1].cuda().long()
#         p = network.predict(x)
#         if p.size(1) == 1:
#             correct += (p.gt(0).eq(y).float()).sum().item()
#         else:
#             correct += (p.argmax(1).eq(y).float()).sum().item()
#         total += len(x)
#         total1 += 1
#     network.train()
#     return correct / total

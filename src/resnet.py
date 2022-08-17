"""
resnet implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet56']

def weight_init(m):
    """
    Init CNN weights
    """
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    """
    Identity mapping between ResNet block with different size feature map
    """
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride !=1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 experiment ResNet uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:\
                                            F.pad(x[:,:,::2,::2], \
                                                  (0,0,0,0,planes//4,planes//4), 
                                                  "constant", 0))
            else:
                """
                Dont need right now
                """
                pass

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, with_clusters=False, num_clusters=10, embed_dim=5):
        super(ResNet, self).__init__()
        self.with_clusters = with_clusters
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # this is the linear layer and needs modification for our setting
        if self.with_clusters:
            self.num_clusters = num_clusters
            self.embeds = nn.Embedding(num_embeddings=self.num_clusters, embedding_dim=embed_dim)
            input_size = 64 + embed_dim - 1
        else:
            input_size = 64
        self.linear = nn.Linear(input_size, num_classes)
        self.apply(weight_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x,y=[]):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        # the linear connect is here again!
        if self.with_clusters:
            cluster_ids = y
            embeds_tensor = self.embeds(cluster_ids.long())
            out = torch.cat((out, embeds_tensor),dim=1)
        out = self.linear(out)
        return out

def resnet20(_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=_classes)

def resnet32(_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=_classes)

def resnet56(_classes=10):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=_classes)

def test(net):
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()

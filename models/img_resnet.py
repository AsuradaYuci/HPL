import torchvision
from torch import nn
from torch.nn import init
from models.utils import pooling
import torch


class ResNet50(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        # 1.Resnet-50
        # resnet50 = torchvision.models.resnet50(pretrained=True)
        resnet50 = torchvision.models.resnet50()
        resnet50.load_state_dict(torch.load('/18640539002/dataset_cc/Pretrain-models/resnet50-19c8e357.pth'))
        if config.MODEL.RES4_STRIDE == 1:
            resnet50.layer4[0].conv2.stride=(1, 1)
            resnet50.layer4[0].downsample[0].stride=(1, 1) 
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        # 2. avg pooling
        if config.MODEL.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.MODEL.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.MODEL.POOLING.NAME == 'gem':
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))
        # 3. BN
        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)
        
    def forward(self, x):
        x = self.base(x)  # torch.Size([32, 2048, 24, 12])
        x = self.globalpooling(x)
        x = x.view(x.size(0), -1)  # 32, 4096
        f = self.bn(x)
        if self.training:
            return x, f
        return f
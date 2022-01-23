import torch
from torch import nn
import torchvision as tv


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        # initial layers , 1X1 convolution
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.batch0 = nn.BatchNorm2d(out_channels)
        # layer1 of ResBLock
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        # layer2 of ResBLock
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, input_tensor):
        # 1X1
        input_changed = self.conv0(input_tensor)
        input_changed = self.batch0(input_changed)
        # first seq
        data = self.conv1(input_tensor)
        data = self.batch1(data)
        data = self.relu1(data)
        # second sequence
        data = self.conv2(data)
        data = self.batch2(data)
        # data = self.relu2(data) : after the skip
        # adding skip connection
        data = data + input_changed
        # then apply relu
        data = self.relu2(data)

        return data


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self layer
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.batch_norm = nn.BatchNorm2d(64)
        self.RelU = nn.ReLU()
        self.max_pool = nn.MaxPool2d(3, 2)
        # resBlock layer
        self.res_block_1 = ResBlock(64, 64, 1)
        self.res_block_2 = ResBlock(64, 128, 2)
        self.res_block_3 = ResBlock(128, 256, 2)
        self.res_block_4 = ResBlock(256, 512, 2)
        # add layer
        self.dropout = nn.Dropout(p=0.5)
        # self layer
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.flattenLayer = nn.Flatten()
        self.fc_layer = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        # initial layers
        data = self.conv(input_tensor)
        data = self.batch_norm(data)
        data = self.RelU(data)
        data = self.max_pool(data)
        # res blocks
        data = self.res_block_1(data)
        data = self.res_block_2(data)
        data = self.res_block_3(data)
        data = self.res_block_4(data)
        #
        data = self.dropout(data)
        # last few
        data = self.global_avg(data)
        data = self.flattenLayer(data)
        # flatten layer,
        # data = data.view(data.size(0), -1)
        data = self.fc_layer(data)
        data = self.sigmoid(data)

        return data

from __future__ import print_function, division
import torch.nn as nn
import torch.utils.data
import torch
from constants import NEW_CLASSES

# code from: https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py#L47

class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding='same', bias=True),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding='same', bias=True),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):

        x = self.conv(x)
        return x



class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super(up_conv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding='same', bias=True),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=NEW_CLASSES, dropout=0.0, softmax = False): #TODO: add dropout
        super(U_Net, self).__init__()
        self.dropout = dropout
        self.softmax = softmax

        n1 = 8
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32, n1 * 64, n1 * 128]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Maxpool7 = nn.MaxPool2d(kernel_size=2, stride=2) 
        # shape is 11 at this point, so if we do another pool, then we get 5, if that is then used for upconv, then we get 10 in stead of 11

        self.Conv1 = conv_block(in_ch, filters[0], self.dropout)
        self.Conv2 = conv_block(filters[0], filters[1], self.dropout)
        self.Conv3 = conv_block(filters[1], filters[2], self.dropout)
        self.Conv4 = conv_block(filters[2], filters[3], self.dropout)
        self.Conv5 = conv_block(filters[3], filters[4], self.dropout)
        self.Conv6 = conv_block(filters[4], filters[5], self.dropout)
        self.Conv7 = conv_block(filters[5], filters[6], self.dropout)
        # self.Conv8 = conv_block(filters[6], filters[7], self.dropout)

        # self.Up8 = up_conv(filters[7], filters[6], self.dropout)
        # self.Up_conv8 = conv_block(filters[7], filters[6], self.dropout)

        self.Up7 = up_conv(filters[6], filters[5], self.dropout)
        self.Up_conv7 = conv_block(filters[6], filters[5], self.dropout)

        self.Up6 = up_conv(filters[5], filters[4], self.dropout)
        self.Up_conv6 = conv_block(filters[5], filters[4], self.dropout)

        self.Up5 = up_conv(filters[4], filters[3], self.dropout)
        self.Up_conv5 = conv_block(filters[4], filters[3], self.dropout)

        self.Up4 = up_conv(filters[3], filters[2], self.dropout)
        self.Up_conv4 = conv_block(filters[3], filters[2], self.dropout)

        self.Up3 = up_conv(filters[2], filters[1], self.dropout)
        self.Up_conv3 = conv_block(filters[2], filters[1], self.dropout)

        self.Up2 = up_conv(filters[1], filters[0], self.dropout)
        self.Up_conv2 = conv_block(filters[1], filters[0], self.dropout)

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.active = torch.nn.Softmax(dim=1)

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        e6 = self.Maxpool5(e5)
        e6 = self.Conv6(e6)

        e7 = self.Maxpool6(e6)
        e7 = self.Conv7(e7)

        # e8 = self.Maxpool7(e7)
        # e8 = self.Conv8(e8)

        # d8 = self.Up8(e8)
        # d8 = torch.nn.functional.pad(d8, (1,0,0,1)) #padding: left, right, top, bottom
        # d8 = torch.cat((e7, d8), dim=1)
        # d8 = self.Up_conv8(d8)

        d7 = self.Up7(e7)
        d7 = torch.cat((e6, d7), dim=1)
        d7 = self.Up_conv7(d7)

        d6 = self.Up6(d7)
        d6 = torch.cat((e5, d6), dim=1)
        d6 = self.Up_conv6(d6)

        d5 = self.Up5(d6)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        if self.softmax:
            out = self.active(out)

        return out
    

class conv_block_old(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super(conv_block_old, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        x = self.conv(x)
        return x
    

class up_conv_old(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super(up_conv_old, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net_old(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=NEW_CLASSES, dropout=0.0): #TODO: add dropout
        super(U_Net_old, self).__init__()
        self.dropout = dropout

        n1 = 8
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32, n1 * 64]#, n1 * 128]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block_old(in_ch, filters[0], self.dropout)
        self.Conv2 = conv_block_old(filters[0], filters[1], self.dropout)
        self.Conv3 = conv_block_old(filters[1], filters[2], self.dropout)
        self.Conv4 = conv_block_old(filters[2], filters[3], self.dropout)
        self.Conv5 = conv_block_old(filters[3], filters[4], self.dropout)
        self.Conv6 = conv_block_old(filters[4], filters[5], self.dropout)
        self.Conv7 = conv_block_old(filters[5], filters[6], self.dropout)

        self.Up7 = up_conv_old(filters[6], filters[5], self.dropout)
        self.Up_conv7 = conv_block_old(filters[6], filters[5], self.dropout)

        self.Up6 = up_conv_old(filters[5], filters[4], self.dropout)
        self.Up_conv6 = conv_block_old(filters[5], filters[4], self.dropout)

        self.Up5 = up_conv_old(filters[4], filters[3], self.dropout)
        self.Up_conv5 = conv_block_old(filters[4], filters[3], self.dropout)

        self.Up4 = up_conv_old(filters[3], filters[2], self.dropout)
        self.Up_conv4 = conv_block_old(filters[3], filters[2], self.dropout)

        self.Up3 = up_conv_old(filters[2], filters[1], self.dropout)
        self.Up_conv3 = conv_block_old(filters[2], filters[1], self.dropout)

        self.Up2 = up_conv_old(filters[1], filters[0], self.dropout)
        self.Up_conv2 = conv_block_old(filters[1], filters[0], self.dropout)

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        e6 = self.Maxpool5(e5)
        e6 = self.Conv6(e6)

        e7 = self.Maxpool6(e6)
        e7 = self.Conv7(e7)

        d7 = self.Up7(e7)
        d7 = torch.cat((e6, d7), dim=1)
        d7 = self.Up_conv7(d7)

        d6 = self.Up6(e6)
        d6 = torch.cat((e5, d6), dim=1)
        d6 = self.Up_conv6(d6)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        d1 = self.active(out)

        return d1
    


# class U_Net(nn.Module):
#     """
#     UNet - Basic Implementation
#     Paper : https://arxiv.org/abs/1505.04597
#     """
#     def __init__(self, in_ch=3, out_ch=15, dropout=0.0): #TODO: add dropout
#         super(U_Net, self).__init__()
#         self.dropout = dropout

#         n1 = 16
#         filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
#         self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.Conv1 = conv_block(in_ch, filters[0], self.dropout)
#         self.Conv2 = conv_block(filters[0], filters[1], self.dropout)
#         self.Conv3 = conv_block(filters[1], filters[2], self.dropout)
#         self.Conv4 = conv_block(filters[2], filters[3], self.dropout)
#         self.Conv5 = conv_block(filters[3], filters[4], self.dropout)

#         self.Up5 = up_conv(filters[4], filters[3], self.dropout)
#         self.Up_conv5 = conv_block(filters[4], filters[3], self.dropout)

#         self.Up4 = up_conv(filters[3], filters[2], self.dropout)
#         self.Up_conv4 = conv_block(filters[3], filters[2], self.dropout)

#         self.Up3 = up_conv(filters[2], filters[1], self.dropout)
#         self.Up_conv3 = conv_block(filters[2], filters[1], self.dropout)

#         self.Up2 = up_conv(filters[1], filters[0], self.dropout)
#         self.Up_conv2 = conv_block(filters[1], filters[0], self.dropout)

#         self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

#         self.active = torch.nn.Sigmoid()

#     def forward(self, x):

#         e1 = self.Conv1(x)

#         e2 = self.Maxpool1(e1)
#         e2 = self.Conv2(e2)

#         e3 = self.Maxpool2(e2)
#         e3 = self.Conv3(e3)

#         e4 = self.Maxpool3(e3)
#         e4 = self.Conv4(e4)

#         e5 = self.Maxpool4(e4)
#         e5 = self.Conv5(e5)

#         d5 = self.Up5(e5)
#         d5 = torch.cat((e4, d5), dim=1)

#         d5 = self.Up_conv5(d5)

#         d4 = self.Up4(d5)
#         d4 = torch.cat((e3, d4), dim=1)
#         d4 = self.Up_conv4(d4)

#         d3 = self.Up3(d4)
#         d3 = torch.cat((e2, d3), dim=1)
#         d3 = self.Up_conv3(d3)

#         d2 = self.Up2(d3)
#         d2 = torch.cat((e1, d2), dim=1)
#         d2 = self.Up_conv2(d2)

#         out = self.Conv(d2)

#         d1 = self.active(out)

#         return d1

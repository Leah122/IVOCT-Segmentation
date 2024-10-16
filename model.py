import torch
from torch import nn

# base code from: https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py


def conv(n_in, n_out, padding=1):
        return [nn.Conv2d(n_in, n_out, kernel_size=3, stride=1, padding=padding, bias=True),
        nn.BatchNorm2d(n_out),
        nn.ReLU(inplace=True)]

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, pooling = True, dropout=None):
        super(conv_block, self).__init__()
        layers = []

        if pooling: 
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers += conv(in_ch, out_ch)
        if dropout:
            layers.append(nn.Dropout(p=dropout))
        layers += conv(in_ch, out_ch)
        self.pool_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.pool_conv(x)
    

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, dropout=None):
        super(up_conv, self).__init__()
        layers = []

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        layers = conv(in_ch * 2, in_ch)
        if dropout:
            layers.append(nn.Dropout(p=dropout))
        layers += conv(in_ch, in_ch)
        self.conv = nn.Sequential(*layers)

    def forward(self, x, skip_connection):
        y = self.upconv(x)
        y = torch.cat([y, skip_connection], dim=1)
        return self.conv(y)


class Model(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()

        self.encoder = nn.ModuleList(
            [
                # Input shape: (704x704x3 @ 1)
                conv_block(3, 32, dropout=None, pooling=False),
                conv_block(32, 64, dropout=dropout, pooling=True),
                conv_block(64, 64, dropout=dropout, pooling=True),
                conv_block(64, 64, dropout=dropout, pooling=True),
                conv_block(64, 128, dropout=dropout, pooling=True),
                # Output shape: (1x88x88 @ 128)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                # Input shape: (1x88x88 @ 128)
                up_conv(128, 64, dropout=dropout),
                up_conv(64, 64, dropout=dropout),
                up_conv(64, 64, dropout=dropout),
                up_conv(64, 32, dropout=dropout),
                # Output shape: (1x704x704 @ 1)
            ]
        )

    def forward(self, image):
        y = image

        contraction_states = []
        for contraction_block in self.encoder:
            y = contraction_block(y)
            contraction_states.append(y)

        for expansion_block, skip_state in zip(
            self.decoder, reversed(contraction_states[:-1])
        ):
            y = expansion_block(y, skip_state)

        return self.segmentation(y)
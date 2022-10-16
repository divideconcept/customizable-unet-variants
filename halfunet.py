import torch
import torch.nn as nn
import torch.nn.functional as F

def block(in_channels, out_channels, conv_per_block, kernel_size, batch_norm):
    sequence = []
    for i in range(conv_per_block):
        sequence.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size,
                                  padding=(kernel_size - 1) // 2))
        if batch_norm:
            sequence.append(nn.BatchNorm2d(out_channels))
        sequence.append(nn.ReLU(inplace=True))
    return nn.Sequential(*sequence)

class Encoder(nn.Module):
    def __init__(self, depth, in_channels, out_channels, conv_per_block, kernel_size, batch_norm):
        super().__init__()
        self.depth = depth
        if depth > 0:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = block(in_channels, out_channels, conv_per_block, kernel_size, batch_norm)

    def forward(self, x):
        if self.depth > 0:
            x = self.pool(x)
        x = self.block(x)
        return x

class HalfUNet(nn.Module):
    """ `HalfUNet` class is based on https://www.frontiersin.org/articles/10.3389/fninf.2022.911679/full

    Default parameters correspond to the original Half UNet (with the Ghost module).

    Args:
        in_channels: number of channels in the input tensor.
        out_channels: number of channels in the output tensor.
        feature_channels: number of channels in the first and last hidden feature layer.
        depth: number of levels
        conv_per_block: number of convolutions per level block
        kernel_size: kernel size for all block convolutions
        batch_norm: use batch norm
        add_merging: merge layers from different levels using a add instead of a concat
    """

    def __init__(self, in_channels=1, out_channels=2, feature_channels=64,
                 depth=5, conv_per_block=3, kernel_size=3, batch_norm=True, add_merging=True):
        super().__init__()

        self.encoders = []
        self.add_merging=add_merging

        # create the encoder pathway and add to a list
        for i in range(depth):
            encoder = Encoder(i, in_channels if i == 0 else feature_channels, feature_channels, conv_per_block, kernel_size, batch_norm)
            self.encoders.append(encoder)
        # add the list of modules to current module
        self.encoders = nn.ModuleList(self.encoders)

        # create the decoder
        self.upsamplers = []
        for i in range(depth):
            self.upsamplers.append(nn.Upsample(mode='bilinear', scale_factor=2 ** i))
        self.upsamplers = nn.ModuleList(self.upsamplers)
        self.final_block=block(feature_channels if add_merging else feature_channels*depth, feature_channels, conv_per_block-1, kernel_size, batch_norm)
        self.final_conv=nn.Conv2d(feature_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        encoder_outs = []

        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            encoder_outs.append(self.upsamplers[i](x))

        if self.add_merging:
            x=encoder_outs[0]
            for i in range(1,len(encoder_outs)):
                x+=encoder_outs[i]
        else:
            x=torch.cat(encoder_outs,1)

        x=self.final_block(x)
        x=self.final_conv(x)
        return x
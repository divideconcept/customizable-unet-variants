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
    def __init__(self, depth, in_channels, out_channels, kernel_size, batch_norm):
        super().__init__()
        self.depth = depth
        if depth > 0:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = block(in_channels, out_channels, 2, kernel_size, batch_norm)

    def forward(self, x):
        if self.depth > 0:
            x = self.pool(x)
        x = self.block(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encoders_channels, decoders_channels, feature_channels, kernel_size, batch_norm):
        super().__init__()
        self.encoders_conv = []
        self.decoders_conv = []

        for i, encoder_channels in enumerate(encoders_channels):
            sequence = []
            sequence.append(nn.MaxPool2d(kernel_size=2**i, stride=2**i))
            sequence.append(nn.Conv2d(encoder_channels, feature_channels, kernel_size, padding=(kernel_size - 1) // 2))
            if batch_norm:
                sequence.append(nn.BatchNorm2d(feature_channels))
            sequence.append(nn.ReLU(inplace=True))
            self.encoders_conv.append(nn.Sequential(*sequence))
        self.encoders_conv = nn.ModuleList(self.encoders_conv)

        for i, decoder_channels in enumerate(decoders_channels):
            sequence = []
            sequence.append(nn.Upsample(mode='bilinear', scale_factor=2 ** (i + 1)))
            sequence.append(nn.Conv2d(decoder_channels, feature_channels, kernel_size, padding=(kernel_size - 1) // 2))
            if batch_norm:
                sequence.append(nn.BatchNorm2d(feature_channels))
            sequence.append(nn.ReLU(inplace=True))
            self.decoders_conv.append(nn.Sequential(*sequence))
        self.decoders_conv = nn.ModuleList(self.decoders_conv)

        cat_channels=feature_channels * (len(encoders_channels) + len(decoders_channels))
        self.block = block(cat_channels, cat_channels, 1, kernel_size, batch_norm)

    def forward(self, encoders, decoders):
        x = []
        for i, encoder in enumerate(encoders):
            x.append(self.encoders_conv[i](encoder))
        for i, decoder in enumerate(decoders):
            x.append(self.decoders_conv[i](decoder))
        x = torch.cat(x, 1)
        x = self.block(x)
        return x


class UNet3P(nn.Module):
    """ `UNet3P` class is based on https://arxiv.org/abs/2004.08790
    UNet3+ is a convolutional encoder-decoder neural network.

    Default parameters correspond to the original UNet3+.

    Args:
        in_channels: number of channels in the input tensor.
        out_channels: number of channels in the output tensor.
        feature_channels: number of channels in the first and last hidden feature layer.
        depth: number of levels
        kernel_size: kernel size for all block convolutions
        batch_norm: use batch norm
    """

    def __init__(self, in_channels=3, out_channels=1, feature_channels=64,
                 depth=5, kernel_size=3, batch_norm=True):
        super().__init__()

        self.encoders = []
        self.decoders = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = in_channels if i == 0 else feature_channels * (2 ** (i - 1))
            outs = feature_channels * (2 ** i)
            encoder = Encoder(i, ins, outs, kernel_size, batch_norm)
            self.encoders.append(encoder)

        # create the decoder pathway and add to a list
        # decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = []
            outs = []
            for j in range(i, -1, -1):
                ins.append(feature_channels * (2 ** j))
            for j in range(depth - i - 2):
                outs.append(feature_channels*depth)
            outs.append(feature_channels * (2 ** (depth - 1)))
            decoder = Decoder(ins, outs, feature_channels, kernel_size, batch_norm)
            self.decoders.append(decoder)

        self.conv_final = nn.Conv2d(feature_channels*depth, out_channels, kernel_size, padding=(kernel_size - 1) // 2)

        # add the list of modules to current module
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x):
        encoder_outs = []
        decoder_outs = [0 for i in range(len(self.decoders))]

        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            encoder_outs.append(x)

        for i in range(len(self.decoders) - 1, -1, -1):
            decoder_outs[i] = self.decoders[i](encoder_outs[:i + 1][::-1], decoder_outs[i + 1:]+[encoder_outs[-1]])

        output=self.conv_final(decoder_outs[0])
        #        for i, decoder in enumerate(self.decoders):
        #            x = decoder(encoder_outs[-(i+2)], x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        #        x = self.conv_final(x)
        return output
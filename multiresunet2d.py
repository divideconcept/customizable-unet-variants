import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()
        self.batch_norm = batch_norm
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, padding=0)
    
        channels_2=out_channels//2
        channels_3=out_channels//3
        channels_6=out_channels-channels_2-channels_3
        
        self.conv3x3 = nn.Conv2d(in_channels, channels_6, 3, padding=1)
        self.conv5x5 = nn.Conv2d(channels_6, channels_3, 3, padding=1)
        self.conv7x7 = nn.Conv2d(channels_3, channels_2, 3, padding=1)
        
        if self.batch_norm:
            self.batch1 = nn.BatchNorm2d(out_channels)
            self.batch2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        shortcut = self.shortcut(x)
        conv3x3 = F.relu(self.conv3x3(x))
        conv5x5 = F.relu(self.conv5x5(conv3x3))
        conv7x7 = F.relu(self.conv7x7(conv5x5))
        
        x = torch.cat([conv3x3, conv5x5, conv7x7], 1)
        if self.batch_norm:
            x = self.batch1(x)
            
        x = F.relu(shortcut+x)
        if self.batch_norm:
            x = self.batch2(x)

        return x
        
class ResPath(nn.Module):
    def __init__(self, channels, length, batch_norm=True):
        super().__init__()
        self.batch_norm = batch_norm
        
        self.shortcut = []
        self.conv = []
        self.batch1 = []
        self.batch2 = []
        
        for i in range(length):
            self.shortcut.append(nn.Conv2d(channels, channels, 1, padding=0))
            self.conv.append(nn.Conv2d(channels, channels, 3, padding=1))
            if self.batch_norm:
                self.batch1.append(nn.BatchNorm2d(channels))
                self.batch2.append(nn.BatchNorm2d(channels))
    
        self.shortcut = nn.ModuleList(self.shortcut)
        self.conv = nn.ModuleList(self.conv)
        self.batch1 = nn.ModuleList(self.batch1)
        self.batch2 = nn.ModuleList(self.batch2)
    def forward(self, x):
        for i in range(len(self.shortcut)):
            shortcut = self.shortcut[i](x)
            x = F.relu(self.conv[i](x))

            if self.batch_norm:
                x = self.batch1[i](x)
            
            x = F.relu(shortcut+x)
            if self.batch_norm:
                x = self.batch2[i](x)

        return x

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm, conv_downscaling, pooling=True):
        super().__init__()
        
        self.pooling = pooling

        self.block = MultiResBlock(in_channels, out_channels, batch_norm)

        if self.pooling:
            if not conv_downscaling:
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            else:
                self.pool = nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=2)

    def forward(self, x):
        x = self.block(x)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm,
                 add_merging, conv_upscaling):
        super().__init__()

        self.add_merging = add_merging
 
        if not conv_upscaling:
            self.upconv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)
        else:
            self.upconv = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=2),
            nn.Conv2d(in_channels, out_channels,kernel_size=1,groups=1,stride=1))

            
        self.block = MultiResBlock(out_channels*2 if not add_merging else out_channels, out_channels, batch_norm)

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        if not self.add_merging:
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = self.block(x)
        return x


class MultiResUNet2D(nn.Module):
    """ `MultiResUNet` class is based on https://arxiv.org/pdf/1902.04049.pdf
    MultiResUNet is a convolutional encoder-decoder neural network.
    
    Default parameters correspond to the original MultiResUNet.

    Args:
        in_channels: number of channels in the input tensor.
        out_channels: number of channels in the output tensor.
        feature_channels: number of channels in the first and last hidden feature layer.
        depth: number of levels
        batch_norm: add a batch norm after ReLU
        conv_upscaling: use a nearest upscale+conv instead of transposed convolution
        conv_downscaling: use a strided convolution instead of maxpooling
        add_merging: merge layers from different levels using a add instead of a concat
    """

    def __init__(self, in_channels=1, out_channels=1, feature_channels=48,
                       depth=5, batch_norm=True,
                       conv_upscaling=False, conv_downscaling=False, add_merging=False):
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.feature_channels = feature_channels
        self.depth = depth

        self.down_convs = []
        self.up_convs = []
        self.paths = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.feature_channels*(2**i)
            pooling = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, batch_norm,
                                conv_downscaling, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, batch_norm,
                            conv_upscaling=conv_upscaling, add_merging=add_merging)
            self.up_convs.append(up_conv)
            res_path = ResPath(outs, i+1, batch_norm)
            self.paths.append(res_path)

        self.conv_final = nn.Conv2d(outs, self.out_channels,kernel_size=1,groups=1,stride=1)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.paths = nn.ModuleList(self.paths)

    def forward(self, x):
        encoder_outs = []
         
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            path=self.paths[i](before_pool)
            x = module(path, x)
        
        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x

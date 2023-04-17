import torch
import torch.nn as nn
import torch.nn.functional as F

#heavily modified from https://github.com/jaxony/unet-pytorch/blob/master/model.py
def block(in_channels, out_channels, conv_per_block, kernel_size, batch_norm=False):
    sequence = []
    for i in range(conv_per_block):
        sequence.append(nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2))
        sequence.append(nn.ReLU(inplace=True))
        if batch_norm:
            #BatchNorm best after ReLU:
            #https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
            #https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout#comment78277697_40295999
            #https://github.com/cvjena/cnn-models/issues/3
            sequence.append(nn.BatchNorm2d(out_channels))    
    return nn.Sequential(*sequence)

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_per_block, kernel_size, batch_norm, conv_downscaling, pooling=True):
        super().__init__()
        
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.conv_per_block=conv_per_block
        self.kernel_size=kernel_size
        self.batch_norm=batch_norm
        self.conv_downscaling=conv_downscaling
        self.pooling = pooling

        self.block = block(in_channels, out_channels, conv_per_block, kernel_size, batch_norm)

        if self.pooling:
            if not conv_downscaling:
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            else:
                self.pool = nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, stride=2)

    def forward(self, x):
        x = self.block(x)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool

    def extra_repr(self):
            # (Optional)Set the extra information about this module. You can test
            # it by printing an object of this class.
            return 'in_channels={}, out_channels={}, conv_per_block={}, kernel_size={}, batch_norm={}, conv_downscaling={}, pooling={}'.format(
                self.in_channels, self.out_channels, self.conv_per_block, self.kernel_size, self.batch_norm, self.conv_downscaling, self.pooling
            )


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_per_block, kernel_size, batch_norm,
                 add_merging, conv_upscaling):
        super().__init__()

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.conv_per_block=conv_per_block
        self.kernel_size=kernel_size
        self.batch_norm=batch_norm
        self.add_merging = add_merging
        self.conv_upscaling = conv_upscaling
 
        if not conv_upscaling:
            self.upconv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)
        else:
            self.upconv = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=2),
            nn.Conv2d(in_channels, out_channels,kernel_size=1,groups=1,stride=1))

            
        self.block = block(out_channels*2 if not add_merging else out_channels, out_channels, conv_per_block, kernel_size, batch_norm)

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        if not self.add_merging:
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = self.block(x)
        return x

    def extra_repr(self):
            # (Optional)Set the extra information about this module. You can test
            # it by printing an object of this class.
            return 'in_channels={}, out_channels={}, conv_per_block={}, kernel_size={}, batch_norm={}, add_merging={}, conv_upscaling={}'.format(
                self.in_channels, self.out_channels, self.conv_per_block, self.kernel_size, self.batch_norm, self.add_merging, self.conv_upscaling
            )

class TransUNet(nn.Module):
    """ `TransUNet` class is based on https://arxiv.org/abs/2102.04306
    TransUNet is a UNet with a Transformer at its core.
    
    Default parameters are similar to the original TransUNet.

    Args:
        in_channels: number of channels in the input tensor.
        out_channels: number of channels in the output tensor.
        feature_channels: number of channels in the first and last hidden feature layer.
        depth: number of levels
        conv_per_block: number of convolutions per level block
        kernel_size: kernel size for all block convolutions
        batch_norm: add a batch norm after ReLU
        conv_upscaling: use a nearest upscale+conv instead of transposed convolution
        conv_downscaling: use a strided convolution instead of maxpooling
        add_merging: merge layers from different levels using a add instead of a concat
        trans_heads: number of transformer heads
        trans_depth: number of transformer layers
        trans_width: dimension of the transformer feedforward layer
    """

    def __init__(self, in_channels=1, out_channels=2, feature_channels=64,
                       depth=4, conv_per_block=2, kernel_size=3, batch_norm=True,
                       conv_upscaling=True, conv_downscaling=False, add_merging=False,
                       trans_heads=12, trans_depth=12, trans_width=2048):
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.feature_channels = feature_channels
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.feature_channels*(2**i)
            pooling = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, conv_per_block, kernel_size, batch_norm,
                                conv_downscaling, pooling=pooling)
            self.down_convs.append(down_conv)
            
        transformer_layer = nn.TransformerEncoderLayer(d_model=self.feature_channels*(2**(depth-1)), nhead=trans_heads, dim_feedforward=trans_width)
        self.transformer=nn.TransformerEncoder(transformer_layer, num_layers=trans_depth)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, conv_per_block, kernel_size, batch_norm,
                            conv_upscaling=conv_upscaling, add_merging=add_merging)
            self.up_convs.append(up_conv)

        self.conv_final = nn.Conv2d(outs, self.out_channels,kernel_size=1,groups=1,stride=1)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

    def forward(self, x):
        encoder_outs = []
         
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        N, C, H, W = x.shape
        x = x.view(N, C, H*W) #NCHW to NES
        x = x.permute(2, 0, 1) #NES to SNE
        x = self.transformer(x)
        x = x.permute(1, 0, 2) #SNE to NES
        x = x.reshape(N, C, H, W) #NES to NCHW

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
        
        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x

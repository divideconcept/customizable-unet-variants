# Customizable UNet Variants

Different variants and evolutions of the UNet model in highly customizable forms.
Requires PyTorch 1.7 or higher.

**UNet**: The original UNet from https://arxiv.org/abs/1505.04597 (2015)
- in_channels: number of channels in the input tensor.
- out_channels: number of channels in the output tensor.
- feature_channels: number of channels in the first and last hidden feature layer.
- depth: number of levels
- conv_per_block: number of convolutions per level block
- kernel_size: kernel size for all block convolutions
- batch_norm: add a batch norm after ReLU
- conv_upscaling: use a nearest upscale+conv instead of transposed convolution
- conv_downscaling: use a strided convolution instead of maxpooling
- add_merging: merge layers from different levels using a add instead of a concat

**WaveUNet**: 1D variant of the UNet from https://arxiv.org/abs/1806.03185 (2018)
- in_channels: number of channels in the input tensor.
- out_channels: number of channels in the output tensor.
- feature_channels: number of channels in the first and last hidden feature layer.
- depth: number of levels
- conv_per_block: number of convolutions per level block
- kernel_size: kernel size for all block convolutions
- batch_norm: add a batch norm after ReLU
- conv_upscaling: use a nearest upscale+conv instead of transposed convolution
- conv_downscaling: use a strided convolution instead of maxpooling
- add_merging: merge layers from different levels using a add instead of a concat

**MultiResUNet**: https://arxiv.org/abs/1902.04049 (2019)
- in_channels: number of channels in the input tensor.
- out_channels: number of channels in the output tensor.
- feature_channels: number of channels in the first and last hidden feature layer.
- depth: number of levels
- batch_norm: add a batch norm after ReLU
- conv_upscaling: use a nearest upscale+conv instead of transposed convolution
- conv_downscaling: use a strided convolution instead of maxpooling
- add_merging: merge layers from different levels using a add instead of a concat

**UNet3+**: https://arxiv.org/abs/2004.08790 (2020)
- in_channels: number of channels in the input tensor.
- out_channels: number of channels in the output tensor.
- feature_channels: number of channels in the first and last hidden feature layer.
- depth: number of levels
- kernel_size: kernel size for all block convolutions
- batch_norm: use batch norm

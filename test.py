import torch
from stylegan2 import ModulatedConv2d

batch = 8
in_channels, out_channels = 16, 64
style_dim = 512
ada_conv = ModulatedConv2d(in_channels, out_channels, 5, style_dim, 6)
style = torch.randn((batch, style_dim))
input = torch.randn((batch, in_channels, 32, 32))
out = ada_conv(input, style)
print(out.shape)

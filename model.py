import math
import random
import torch
from torch import nn
from torch.nn import functional as F
from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix
from layers import (
    PixelNorm, make_kernel, Upsample, Downsample, Blur, EqualConv2d,
    ModulatedConv2d, EqualLinear, NoiseInjection,
    SelfAttention, CrossAttention, TextEncoder,
)

class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out

def append_if(condition, var, elem):
    if (condition):
        var.append(elem)

class Generator(nn.Module):
    def __init__(
        self, size, style_dim, n_mlp, tin_dim, tout_dim,
        channel_multiplier=2, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size
        style_dim = style_dim + tout_dim
        self.style_dim = style_dim
        self.text_encoder = TextEncoder(tin_dim, tout_dim)

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.attns.append(SelfAttention(out_channel, style_dim))
            self.attns.append(CrossAttention(out_channel, tout_dim))

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self, styles, text_embeds,
        return_latents=False, inject_index=None, truncation=1, truncation_latent=None,
        input_is_latent=False, noise=None, randomize_noise=True, return_images=True,
    ):
        seq_len = text_embeds.shape[1]
        text_embeds = self.text_encoder(text_embeds)
        t_local, t_global = torch.split(text_embeds, [seq_len-1, 1], dim=1)
        # batch, tout_dim
        t_global = t_global.squeeze(dim=1)
        styles = [torch.cat([styles, t_global], dim=1)]

        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        images = []
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])
        append_if(return_images, images, skip)

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb, self_attn, cross_attn in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs,
            self.attns[::2], self.attns[1::2],
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            out = self_attn(out, latent[:, i + 1])
            out = cross_attn(out, t_local)
            skip = to_rgb(out, latent[:, i + 2], skip)
            append_if(return_images, images, skip)

            i += 2

        if not return_images:
            images = [skip]

        if return_latents:
            return images, latent
        else:
            return images, None


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out

class Predictor(nn.Module):
    def __init__(self, in_channel, tin_dim):
        super().__init__()
        layers = []
        inc, outc = in_channel, 256
        for i in range(4):
            layers.append(nn.Conv2d(inc, outc, 1))
            layers.append(nn.GELU())
            inc = outc
        self.conv1 = nn.Sequential(*layers)
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.V = nn.Linear(tin_dim, 256)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, 256, 1),
            nn.GELU(),
        )

    def forward(self, image_embeds, text_embeds):
        # [n, c, h, w] --> [n, 256, h, w]
        out1 = self.conv1(image_embeds)
        # [n, 256]
        out1 = self.global_avg(out1).squeeze()
        # [n, tin_dim] --> [n, 256]
        out2 = self.V(text_embeds)
        # [n, 256] x [n, 256]
        batch = image_embeds.shape[0]
        if batch != out2.shape[0]:
            out2 = out2.repeat(batch // out2.shape[0], 1)
        out = torch.mul(out1, out2).sum(dim=1, keepdim=True)
        # [n, c, h, w] --> [n, 256, h, w]
        out3 = self.conv2(image_embeds)
        # [n, 256, h, w] --> [n, 256]
        out3 = self.global_avg(out3).squeeze()
        # [n, 256] --> [n, 1]
        out = (out + out3).sum(dim=1, keepdim=True)
        return out

class Discriminator(nn.Module):
    def __init__(self, size, tin_dim, tout_dim, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.heads = nn.ModuleList([ConvLayer(3, channels[size], 1)])
        self.convs = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.predictors = nn.ModuleList()
        log_size = int(math.log(size, 2))

        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            self.convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            self.attns.append(SelfAttention(out_channel))
            self.heads.append(ConvLayer(3, in_channel, 1))
            in_channel = out_channel
            self.predictors.append(Predictor(in_channel, tout_dim))
        self.text_encoder = TextEncoder(tin_dim, tout_dim)

    def forward(self, inputs, text_embeds):
        batch = text_embeds.shape[0]
        # [n, seq_len, tin_dim] --> [n, tout_dim]
        text_embeds = self.text_encoder(text_embeds)[:, -1]
        # inputs: 4x --> 8x --> ... --> 64x
        in_len = len(inputs)
        # 64x --> 32x --> ... --> 4x
        input_heads = []
        for i in range(len(inputs)):
            input = inputs[in_len-i-1]
            input_heads.append(self.heads[i](input))

        outputs, i = [], 0
        for conv, attn in zip(self.convs, self.attns):
            input = input_heads[i]
            if len(outputs):
                input = torch.cat([input, outputs[-1]], dim=0)
            out = conv(input)
            out = attn(out)
            outputs.append(out)
            i += 1

        preds = []
        for out, predictor in zip(outputs, self.predictors):
            preds.append(predictor(out, text_embeds).view(batch, -1))

        out = torch.cat(preds, dim=1)
        return out


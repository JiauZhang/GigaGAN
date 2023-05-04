import math
import random
import torch
from torch import nn
from torch.nn import functional as F
from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix
from layers import (
    PixelNorm, ToRGB, ConstantInput, StyledConv, Blur, EqualConv2d,
    ModulatedConv2d, EqualLinear, NoiseInjection,
    SelfAttention, CrossAttention, TextEncoder,
)

def append_if(condition, var, elem):
    if (condition):
        var.append(elem)

class Generator(nn.Module):
    def __init__(
        self, size, z_dim, n_mlp, tin_dim=0, tout_dim=0,
        channel_multiplier=2, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01,
        use_self_attn=False, use_text_cond=False, use_multi_scale=False,
        attn_res=[8, 16, 32],
    ):
        super().__init__()

        self.size = size
        self.use_multi_scale = use_multi_scale
        self.use_self_attn = use_self_attn
        self.use_text_cond = use_text_cond
        if use_text_cond:
            self.style_dim = z_dim + tout_dim
            self.text_encoder = TextEncoder(tin_dim, tout_dim)
        else:
            self.style_dim = z_dim

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    self.style_dim, self.style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512, 8: 512, 16: 512, 32: 512, 64: 256 * channel_multiplier,
            128: 128 * channel_multiplier, 256: 64 * channel_multiplier,
            512: 32 * channel_multiplier, 1024: 16 * channel_multiplier,
        }
        n_kernels = {
            4: 1, 8: 1, 16: 2, 32: 4, 64: 8,
            128: 8, 256: 8, 512: 8, 1024: 8,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, self.style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], self.style_dim, upsample=False)

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
            res = 2 ** i
            out_channel = self.channels[res]

            self.convs.append(StyledConv(
                in_channel, out_channel, 3, self.style_dim, upsample=True,
                blur_kernel=blur_kernel, n_kernel=n_kernels[res],
            ))
            self.convs.append(StyledConv(
                out_channel, out_channel, 3, self.style_dim, blur_kernel=blur_kernel,
                n_kernel=n_kernels[res],
            ))

            self.attns.append(
                SelfAttention(out_channel) if use_self_attn and res in attn_res else None
            )
            self.attns.append(
                CrossAttention(out_channel, tout_dim) if use_text_cond and res in attn_res else None
            )

            self.to_rgbs.append(ToRGB(out_channel, self.style_dim))

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
        self, styles, text_embeds=None,
        return_latents=False, inject_index=None, truncation=1, truncation_latent=None,
        input_is_latent=False, noise=None, randomize_noise=True,
    ):
        if self.use_text_cond:
            seq_len = text_embeds.shape[1]
            text_embeds = self.text_encoder(text_embeds)
            t_local, t_global = torch.split(text_embeds, [seq_len-1, 1], dim=1)
            # batch, tout_dim
            t_global = t_global.squeeze(dim=1)
            styles = [torch.cat([style_, t_global], dim=1) for style_ in styles]

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
        append_if(self.use_multi_scale, images, skip)

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb, self_attn, cross_attn in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs,
            self.attns[::2], self.attns[1::2],
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            out = self_attn(out) if self_attn else out
            out = cross_attn(out, t_local) if cross_attn else out
            skip = to_rgb(out, latent[:, i + 2], skip)
            append_if(self.use_multi_scale, images, skip)

            i += 2

        # images: [4x, 8x, ..., 32x, 64x] or [64x]
        if not self.use_multi_scale:
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
    def __init__(self, size, tin_dim=0, tout_dim=0, channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1], use_multi_scale=False, use_self_attn=False,
        use_text_cond=False,
    ):
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

        self.use_multi_scale = use_multi_scale
        self.use_self_attn = use_self_attn
        self.use_text_cond = use_text_cond
        self.heads = nn.ModuleList()
        self.convs = nn.ModuleList([ConvLayer(3, channels[size], 1)])
        self.attns = nn.ModuleList([None])
        self.predictors = nn.ModuleList()
        log_size = int(math.log(size, 2))

        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            self.convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            self.attns.append(SelfAttention(out_channel) if use_multi_scale else None)
            self.heads.append(ConvLayer(3, in_channel, 1) if use_multi_scale else None)
            in_channel = out_channel
            self.predictors.append(Predictor(in_channel, tout_dim) if use_multi_scale else None)
        self.text_encoder = TextEncoder(tin_dim, tout_dim) if use_text_cond else None

    def forward(self, inputs, text_embeds=None):
        if self.use_text_cond:
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
        out = inputs[-1]
        for conv, attn in zip(self.convs, self.attns):
            out = conv(out)
            out = attn(out) if attn else out
            i += 1
        out = torch.mean(out, dim=[1, 2, 3])
        return out


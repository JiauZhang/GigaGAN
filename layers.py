import torch, math
from torch import nn
from torch.nn import functional as F
from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix

class AdaWeight(nn.Module):
    def __init__(self, n_kernel, in_channels, out_channels, kernel_size,
        style_dim=512,
    ):
        super().__init__()
        self.n_kernel = n_kernel
        # conv weight shape: out_ch, in_ch, k_h, k_w
        self.weight = nn.Parameter(torch.empty((n_kernel, out_channels, in_channels, kernel_size, kernel_size)))
        self.reset_parameters()
        self.ada_weight = nn.Linear(style_dim, n_kernel)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, style):
        # batch, n_kernel
        ada_weight = self.ada_weight(style).softmax(dim=-1)
        ada_weight = ada_weight.view(-1, self.n_kernel, 1, 1, 1, 1)
        weight = (ada_weight * self.weight).sum(dim=1)
        return weight

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k

class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)
        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2
        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)
        return out

class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)
        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2
        pad1 = p // 2
        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)
        return out

class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)
        self.register_buffer("kernel", kernel)
        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        return out

# https://github.com/facebookresearch/semi-discrete-flow/blob/main/model_itemsets.py#L306
class L2MultiheadAttention(nn.Module):
    """ Kim et al. "The Lipschitz Constant of Self-Attention" https://arxiv.org/abs/2006.04710 """
    def __init__(self, embed_dim, num_heads):
        super(L2MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_weight = nn.Parameter(torch.empty(embed_dim, num_heads, self.head_dim))
        self.v_weight = nn.Parameter(torch.empty(embed_dim, num_heads, self.head_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_weight.reshape(self.embed_dim, self.embed_dim))
        nn.init.xavier_uniform_(self.v_weight.reshape(self.embed_dim, self.embed_dim))

    def forward(self, x):
        """
        Args:
            x: (T, N, D)
            attn_mask: (T, T) added to pre-softmax logits.
        """

        T, N, _ = x.shape

        q = torch.einsum("tbm,mhd->tbhd", x, self.q_weight)
        k = torch.einsum("tbm,mhd->tbhd", x, self.q_weight)
        squared_dist = (
            torch.einsum("tbhd,tbhd->tbh", q, q).unsqueeze(1)
            + torch.einsum("sbhd,sbhd->sbh", k, k).unsqueeze(0)
            - 2 * torch.einsum("tbhd,sbhd->tsbh", q, k)
        )
        attn_logits = -squared_dist / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_logits, dim=1)  # (T, S, N, H)
        A = torch.einsum("mhd,nhd->hmn", self.q_weight, self.q_weight) / math.sqrt(
            self.head_dim
        )
        XA = torch.einsum("tbm,hmn->tbhn", x, A)
        PXA = torch.einsum("tsbh,sbhm->tbhm", attn_weights, XA)
        PXAV = torch.einsum("tbhm,mhd->tbhd", PXA, self.v_weight).reshape(
            T, N, self.embed_dim
        )
        return self.out_proj(PXAV)

class SelfAttention(nn.Module):
    def __init__(self, in_channels, style_dim, num_heads=8):
        super().__init__()

        self.embedding = nn.Linear(style_dim, in_channels)
        self.in_channels = in_channels
        self.l2attn = L2MultiheadAttention(in_channels, num_heads)
        self.ff = nn.Sequential(
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        )
        self.ln1 = nn.LayerNorm(in_channels)
        self.ln2 = nn.LayerNorm(in_channels)

    def forward(self, input, style):
        batch, c, h, w = input.shape
        # style: [N, style_dim] --> [N, C]
        style_embed = self.embedding(style)
        # input: [N, C, H, W] --> [N, H, W, C] --> [N, HWC]
        input = input.permute(0, 2, 3, 1).reshape(batch, h*w*c)
        # [N, HWC+C] --> [N, HW+1, C]
        input_stype = torch.cat([input, style_embed], dim=-1).reshape(batch, h*w+1, c)
        # [N, HW+1, C]
        out1 = self.l2attn(input_stype)
        out1 = self.ln1(out1 + input_stype)
        out2 = self.ff(out1.view(batch*(h*w+1), c)).view(batch, h*w+1, c)
        output = self.ln2(out2 + out1)
        # [N, HW, C]
        output = torch.split(output, [h*w, 1], dim=1)[0]
        output = output.reshape(batch, h, w, c).permute(0, 3, 1, 2)
        return output

class TextEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=8):
        super().__init__()

        self.embedding = nn.Linear(in_dim, out_dim)
        self.l2attn = L2MultiheadAttention(out_dim, num_heads)
        self.ff = nn.Sequential(
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.ln1 = nn.LayerNorm(out_dim)
        self.ln2 = nn.LayerNorm(out_dim)

    def forward(self, text_embeds):
        text_embeds = self.embedding(text_embeds)
        out1 = self.l2attn(text_embeds)
        out1 = self.ln1(out1 + text_embeds)
        out2 = self.ff(out1)
        output = self.ln2(out2 + out1)
        return output

class CrossAttention(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads=8, bias=False):
        super().__init__()

        self.in_channels = in_channels
        self.to_q = nn.Linear(in_channels, in_channels, bias=bias)
        self.to_k = nn.Linear(embed_dim, in_channels, bias=bias)
        self.to_v = nn.Linear(embed_dim, in_channels, bias=bias)
        self.mha = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        )
        self.ln1 = nn.LayerNorm(in_channels)
        self.ln2 = nn.LayerNorm(in_channels)

    def forward(self, image_embeds, text_embeds):
        batch, c, h, w = image_embeds.shape
        # image_embeds: [N, C, H, W] --> [N, H, W, C] --> [N, HW, C]
        image_embeds = image_embeds.permute(0, 2, 3, 1).reshape(batch, h*w, c)
        query = self.to_q(image_embeds)
        key = self.to_k(text_embeds)
        value = self.to_v(text_embeds)
        attn_output, attn_output_weights = self.mha(query, key, value)
        out1 = self.ln1(attn_output + image_embeds)
        out2 = self.ff(out1)
        # [N, HW, C]
        output = self.ln2(out2 + out1)
        output = output.reshape(batch, h, w, c).permute(0, 3, 1, 2)
        return output

class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True,
        n_kernel=8,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )

class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        n_kernel=8,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.n_kernel = n_kernel

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2
        self.ada_weight = AdaWeight(n_kernel, in_channel, out_channel, kernel_size, style_dim=style_dim)
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        self.weight = self.ada_weight(style)

        if not self.fused:
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style)

            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channel, 1, 1)

            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(
                    input, weight, padding=0, stride=2
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)

            else:
                out = conv2d_gradfix.conv2d(input, weight, padding=self.padding)

            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)

            return out

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.reshape(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.reshape(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise

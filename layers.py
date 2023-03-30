import torch, math
from torch import nn
from torch.nn import functional as F
from op import upfirdn2d

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
        nn.init.xavier_uniform_(self.q_weight.view(self.embed_dim, self.embed_dim))
        nn.init.xavier_uniform_(self.v_weight.view(self.embed_dim, self.embed_dim))

    def forward(self, x, y=None):
        """
        Args:
            x: (T, N, D)
            attn_mask: (T, T) added to pre-softmax logits.
        """

        T, N, _ = x.shape
        if y is None:
            y = x
        q = torch.einsum("tbm,mhd->tbhd", x, self.q_weight)
        k = torch.einsum("tbm,mhd->tbhd", y, self.q_weight)
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
        XA = torch.einsum("tbm,hmn->tbhn", y, A)
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
        batch, h, w, c = input.shape
        # style: [N, style_dim] --> [N, C]
        style_embed = self.embedding(style)
        # input: [N, C, H, W] --> [N, H, W, C] --> [N, HWC]
        input = input.permute(0, 2, 3, 1).view(batch, h*w*c)
        # [N, HWC+C] --> [N, HW+1, C]
        input_stype = torch.cat([input, style_embed], dim=-1).view(batch, h*w+1, c)
        # [N, HW+1, C]
        out1 = self.l2attn(input_stype)
        out1 = self.ln1(out1 + input_stype)
        out2 = self.ff(out1.veiw(batch*(h*w+1), c)).view(batch, h*w+1, c)
        output = self.ln2(out2 + out1)
        # [N, HW, C]
        output = torch.split(output, [h*w, 1], dim=1)[0]
        output = output.view(batch, h, w, c).permute(0, 3, 1, 2)
        return output

class TextEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8):
        super().__init__()

        self.embedding = nn.Linear(input_dim, output_dim)
        self.l2attn = L2MultiheadAttention(output_dim, num_heads)
        self.ff = nn.Sequential(
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        self.ln1 = nn.LayerNorm(output_dim)
        self.ln2 = nn.LayerNorm(output_dim)

    def forward(self, text_embeds):
        # text_embeds: [batch, seq_len, embed_dim]
        seq_len = text_embeds.shape[1]
        out1 = self.l2attn(text_embeds)
        out1 = self.ln1(out1 + text_embeds)
        out2 = self.ff(out1)
        output = self.ln2(out2 + out1)
        return torch.split(output, [seq_len-1, 1], dim=1)

class CrossAttention(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads=8):
        super().__init__()

        self.embedding = nn.Linear(embed_dim, in_channels)
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

    def forward(self, image_embeds, text_embeds):
        batch, h, w, c = image_embeds.shape
        # text_embeds: [N, seq_len, embed_dim] --> [N, seq_len, in_channels]
        text_embeds = self.embedding(text_embeds)
        # image_embeds: [N, C, H, W] --> [N, H, W, C] --> [N, HW, C]
        image_embeds = image_embeds.permute(0, 2, 3, 1).view(batch, h*w, c)
        out1 = self.l2attn(image_embeds, text_embeds)
        out1 = self.ln1(out1 + image_embeds)
        out2 = self.ff(out1)
        # [N, HW, C]
        output = self.ln2(out2 + out1)
        output = output.view(batch, h, w, c).permute(0, 3, 1, 2)
        return output

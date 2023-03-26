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

    def forward(self, x, attn_mask=None, rm_nonself_grads=False):
        """
        Args:
            x: (T, N, D)
            attn_mask: (T, T) added to pre-softmax logits.
        """

        T, N, _ = x.shape

        q = k = torch.einsum("tbm,mhd->tbhd", x, self.q_weight)
        squared_dist = (
            torch.einsum("tbhd,tbhd->tbh", q, q).unsqueeze(1)
            + torch.einsum("sbhd,sbhd->sbh", k, k).unsqueeze(0)
            - 2 * torch.einsum("tbhd,sbhd->tsbh", q, k)
        )
        attn_logits = -squared_dist / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_mask = attn_mask[..., None, None]
            attn_logits += attn_mask
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
    def __init__(self, in_channels, style_dim, num_heads=4):
        super().__init__()

        self.embedding = nn.Linear(style_dim, in_channels)
        self.in_channels = in_channels
        self.l2attn = L2MultiheadAttention(in_channels, num_heads)

    def forward(self, input, style):
        batch, h, w = input.shape[:3]
        # style: [N, style_dim] --> [N, C]
        style_embed = self.embedding(style)
        # input: [N, C, H, W] --> [N, H, W, C] --> [N, HWC]
        input = input.permute(0, 2, 3, 1).view(batch, -1)
        # [N, HWC+C] --> [N, HW+1, C]
        input_stype = torch.cat([input, style_embed], dim=-1).view(batch, -1, self.in_channels)
        # [N, HW+1, C]
        output = torch.split(self.l2attn(input_stype), [h*w, 1], dim=1)[0]
        output = output.view(batch, h, w, -1).permute(0, 3, 1, 2)
        return output

class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, text):
        pass

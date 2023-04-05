import torch
from model import Generator, Discriminator
from torchvision import utils

style_dim = 256
tin_dim = 768
tout_dim = 256
image_size = 64
batch = 1
n_mlp = 8
seq_len = 18

with torch.no_grad():
    g = Generator(image_size, style_dim, n_mlp, tin_dim, tout_dim)
    z = torch.randn(batch, style_dim)
    text_embeds = torch.randn(batch, seq_len, tin_dim)
    images = g(z, text_embeds)[0]
    for i in range(len(images)):
        print(images[i].shape)
        utils.save_image(images[i], f"test-{i}.png", nrow=1, normalize=True, value_range=(-1, 1))

    d = Discriminator(image_size)
    out = d(images)

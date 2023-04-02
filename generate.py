import torch
from model import Generator
from torchvision import utils

style_dim = 256
image_size = 64
batch = 1
n_mlp = 8

g = Generator(image_size, style_dim, n_mlp)
z = torch.randn(batch, style_dim)
image = g([z])[0]
utils.save_image(image, "test.png", nrow=1, normalize=True, value_range=(-1, 1))

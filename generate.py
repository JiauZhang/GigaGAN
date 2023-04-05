import torch
from model import Generator
from torchvision import utils

style_dim = 256
text_dim = 256
image_size = 64
batch = 1
n_mlp = 8

g = Generator(image_size, style_dim, n_mlp, text_dim)
z = torch.randn(batch, style_dim)
images = g([z])[0]
for i in range(len(images)):
    print(images[i].shape)
    utils.save_image(images[i], f"test-{i}.png", nrow=1, normalize=True, value_range=(-1, 1))

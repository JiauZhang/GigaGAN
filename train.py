import argparse, math, random, os, torch
import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from torchvision import transforms, utils
from tqdm import tqdm
from op import conv2d_gradfix
from model import Generator, Discriminator
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
from clip import CLIPText

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def multi_scale(image):
    device = image.device
    images = []
    for size in [4, 8, 16, 32]:
        images.append(resize(image, (size, size)).detach().to(device))
    images.append(image.detach())
    return images

def train(args, loader, generator, discriminator, text_encoder, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)
    pbar = tqdm(range(args.iter))

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    loss_dict = {}

    g_module = generator
    d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    sample_z = torch.randn(args.n_sample, args.latent, device=device)
    text = next(loader)['text'][:1]
    print(text)
    if args.use_text_cond:
        sample_t = text_encoder(text)
        sample_t = sample_t.repeat(args.n_sample, 1, 1).detach()
    else:
        sample_t = None

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        image_text = next(loader)
        # [4x, 8x, ..., 64x] or [64x]
        real_img = multi_scale(image_text['image']) if args.use_multi_scale else [image_text['image']]
        real_img = [img.to(args.device) for img in real_img]
        text_embeds = text_encoder(image_text['text']) if args.use_text_cond else None

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, -1, device)
        # fake_img: [4x, 8x, ..., 64x] or [64x]
        fake_img, _ = generator(noise, text_embeds)

        # [batch, 10]
        fake_pred = discriminator(fake_img, text_embeds)
        real_pred = discriminator(real_img, text_embeds)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0
        if d_regularize:
            real_img[-1].requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img[-1])
            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
            d_optim.step()
        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, -1, device)
        fake_img, _ = generator(noise, text_embeds)

        fake_pred = discriminator(fake_img, text_embeds)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        accumulate(g_ema, g_module, accum)

        loss_reduced = loss_dict
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()

        pbar.set_description(
            f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
            f"real_socre: {real_score_val:.4f}; fake_score: {fake_score_val:.4f}; "
            f"augment: {ada_aug_p:.4f}"
        )

        if i % 100 == 0:
            with torch.no_grad():
                g_ema.eval()
                sample, _ = g_ema([sample_z], sample_t)
                utils.save_image(
                    sample[-1], f"sample/{str(i).zfill(6)}.png",
                    nrow=int(math.sqrt(args.n_sample)), normalize=True, value_range=(-1, 1),
                )

        if i % 10000 == 0:
            torch.save(
                {
                    "g": g_module.state_dict(),
                    "d": d_module.state_dict(),
                    "g_ema": g_ema.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "args": args,
                    "ada_aug_p": ada_aug_p,
                },
                f"checkpoint/{str(i).zfill(6)}.pt",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GigaGAN trainer")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--path", type=str, default="lambdalabs/pokemon-blip-captions")
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=64, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.0025, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8
    args.start_iter = 0
    args.tin_dim = 0
    args.tout_dim = 0
    args.use_multi_scale = False
    args.use_text_cond = False
    args.g_use_self_attn = False
    args.d_use_self_attn = False

    device = args.device
    generator = Generator(
        args.size, args.latent, args.n_mlp, args.tin_dim, args.tout_dim,
        channel_multiplier=args.channel_multiplier, use_multi_scale=args.use_multi_scale,
        use_text_cond=args.use_text_cond, use_self_attn=args.g_use_self_attn,
    ).to(device)
    discriminator = Discriminator(
        args.size, args.tin_dim, args.tout_dim, channel_multiplier=args.channel_multiplier,
        use_multi_scale=args.use_multi_scale, use_self_attn=args.d_use_self_attn,
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, args.tin_dim, args.tout_dim,
        channel_multiplier=args.channel_multiplier, use_multi_scale=args.use_multi_scale,
        use_text_cond=args.use_text_cond, use_self_attn=args.g_use_self_attn,
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = generator.set_optim(lr=args.lr)
    d_optim = optim.AdamW(discriminator.parameters(), lr=args.lr, betas=(0, 0.99))

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    to_tensor = transforms.Compose([
        transforms.Resize(args.size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ])
    def preprocess(data):
        for i in range(len(data['image'])):
            data['image'][i] = to_tensor(data['image'][i])
        return data
    dataset = load_dataset('lambdalabs/pokemon-blip-captions', split="train", cache_dir='.')
    dataset = dataset.with_transform(preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)
    text_encoder = CLIPText(args.device) if args.use_text_cond else None

    train(args, dataloader, generator, discriminator, text_encoder, g_optim, d_optim, g_ema, device)

"""CycleGAN networks: ResNet generator, 70x70 PatchGAN discriminator, image pool."""
import random

import torch
import torch.nn as nn


def init_weights(net, std=0.02):
    """Normal(0, std) init for conv / affine-norm weights, zero biases. Returns net."""
    def _init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.InstanceNorm2d) and m.affine:
            nn.init.normal_(m.weight, 1.0, std)
            nn.init.zeros_(m.bias)
    net.apply(_init)
    return net


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """c7s1-ngf, d(2ngf), d(4ngf), n_blocks x R(4ngf), u(2ngf), u(ngf), c7s1-3, Tanh.

    upsample selects the two upsampling stages (everything else is identical):
      "convtranspose" (default, run01): ConvTranspose2d(k3, s2, p1, op1) + InstanceNorm + ReLU, as in the original CycleGAN.
      "resize_conv": Upsample(x2, nearest) + ReflectionPad2d(1) + Conv2d(k3) + InstanceNorm + ReLU, which avoids
                     the checkerboard pattern that transposed convolutions can produce. Same parameter count.
    """

    def __init__(self, ngf=64, n_blocks=6, in_ch=3, out_ch=3, upsample="convtranspose"):
        super().__init__()
        if upsample not in ("convtranspose", "resize_conv"):
            raise ValueError(f"upsample must be 'convtranspose' or 'resize_conv', got {upsample!r}")
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_ch, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        mult = 1
        for _ in range(2):  # downsample x2, twice
            layers += [nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                       nn.InstanceNorm2d(ngf * mult * 2), nn.ReLU(True)]
            mult *= 2
        layers += [ResnetBlock(ngf * mult) for _ in range(n_blocks)]
        for _ in range(2):  # upsample x2, twice
            if upsample == "convtranspose":
                layers += [nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, stride=2, padding=1, output_padding=1)]
            else:  # resize_conv
                layers += [nn.Upsample(scale_factor=2, mode="nearest"), nn.ReflectionPad2d(1),
                           nn.Conv2d(ngf * mult, ngf * mult // 2, 3)]
            layers += [nn.InstanceNorm2d(ngf * mult // 2), nn.ReLU(True)]
            mult //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_ch, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN: C64-C128-C256-C512 (4x4 convs), then a 1-channel patch map.

    No norm on the first layer; InstanceNorm from the second layer on.
    """

    def __init__(self, ndf=64, in_ch=3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for i in range(1, 4):
            prev, mult = mult, min(2 ** i, 8)
            stride = 2 if i < 3 else 1
            layers += [nn.Conv2d(ndf * prev, ndf * mult, 4, stride=stride, padding=1),
                       nn.InstanceNorm2d(ndf * mult), nn.LeakyReLU(0.2, True)]
        layers.append(nn.Conv2d(ndf * mult, 1, 4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ImagePool:
    """History buffer of generated images (Shrivastava et al.) for discriminator updates.

    While not full, store and return the new image. Once full, with prob 0.5 return a
    random stored image (replacing it with the new one), else return the new image.
    """

    def __init__(self, size=50):
        self.size, self.images = size, []

    def query(self, images):
        if self.size == 0:
            return images
        out = []
        for img in images.detach():
            img = img.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(img)
                out.append(img)
            elif random.random() > 0.5:
                idx = random.randrange(self.size)
                out.append(self.images[idx].clone())
                self.images[idx] = img
            else:
                out.append(img)
        return torch.cat(out, 0)


def count_params(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    G = init_weights(ResnetGenerator())
    D = init_weights(PatchDiscriminator())
    x = torch.randn(1, 3, 64, 64)
    y, p = G(x), D(x)
    print(f"G: {tuple(x.shape)} -> {tuple(y.shape)}, params={count_params(G):,}")
    print(f"D: {tuple(x.shape)} -> {tuple(p.shape)}, params={count_params(D):,}")

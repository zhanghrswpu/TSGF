import torch.nn as nn
import random
import torch.fft as fft
import torch
import kornia.augmentation as Kg
from torchvision import transforms as T
from PIL import ImageFilter
import matplotlib.pyplot as plt

def random_transform(X):
    s = torch.randint(0, 2, (1,)).item()
    if s == 0:
        trans_s = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([ImageSplitterAndShuffler()], p=1)
        ])
        return trans_s(X)
    if s == 1:
        Augmentation = nn.Sequential(
            Kg.RandomResizedCrop(size=(224, 224)),
            Kg.RandomHorizontalFlip(p=0.5),
            Kg.ColorJitter(brightness=0.4, contrast=0.8, saturation=0.8, hue=0.2, p=0.8),
            Kg.RandomGrayscale(p=0.2),
            Kg.RandomGaussianBlur((int(0.1 * 224 - 1), int(0.1 * 224 - 1)), (0.1, 2.0), p=0.5),
            T.RandomApply([SpectralPooling()], p=0.2),
            T.RandomApply([ImageSplitterAndShuffler()], p=1)
        )
        return Augmentation(X)
    else:
        return X

class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class SpectralPooling(object):

    def __init__(self, limit=0.05):
        self.limit = limit

    def __call__(self, x):
        pass1 = torch.abs(fft.rfftfreq(x.shape[-1])) < self.limit
        pass1 = pass1.to(x.device)

        pass2 = torch.abs(fft.fftfreq(x.shape[-2])) < self.limit
        pass2 = pass2.to(x.device)

        kernel = torch.outer(pass2, pass1)

        fft_input = fft.rfft2(x).to(x.device)
        return fft.irfft2(fft_input * kernel, s=x.shape[-2:])

class ImageSplitterAndShuffler(object):
    def __init__(self):
        pass

    def __call__(self, images):
        """
        shuffle
        """
        batch, channels, height, width = images.size()

        images_ = torch.zeros_like(images)

        block_height = height // 2
        block_width = width // 2

        block1 = images[:, :, :block_height, :block_width]
        block2 = images[:, :, :block_height, block_width:]
        block3 = images[:, :, block_height:, :block_width]
        block4 = images[:, :, block_height:, block_width:]

        blocks = [block1, block2, block3, block4]
        random.shuffle(blocks)

        images_[:, :, :block_height, :block_width] = blocks[0]
        images_[:, :, :block_height, block_width:] = blocks[1]
        images_[:, :,  block_height:, :block_width] = blocks[2]
        images_[:, :,  block_height:, block_width:] = blocks[3]

        return images_



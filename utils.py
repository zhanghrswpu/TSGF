import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
import lmdb
import pickle
import six
import random
import torch.fft as fft
import os
from torchvision.utils import save_image
import torch
from PIL import ImageFilter

import os
from PIL import Image
from dct import *

class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class SpectralPooling(object):

    def __init__(self, limit=0.05, high_freq_boost=1.2):
        self.limit = limit
        self.high_freq_boost = high_freq_boost

    def __call__(self, x):
        # 获取频率轴
        pass1 = torch.abs(fft.rfftfreq(x.shape[-1])) < self.limit
        pass1 = pass1.to(x.device)

        pass2 = torch.abs(fft.fftfreq(x.shape[-2])) < self.limit
        pass2 = pass2.to(x.device)

        kernel = torch.outer(pass2, pass1)

        # 计算FFT并应用频域滤波器
        fft_input = fft.rfft2(x).to(x.device)
        filtered_fft = fft_input * kernel

        # 增强高频成分
        high_freq_kernel = torch.logical_not(kernel)
        enhanced_fft = fft_input * high_freq_kernel * self.high_freq_boost + filtered_fft

        # 返回逆FFT的结果
        return fft.irfft2(enhanced_fft, s=x.shape[-2:])

class DCT_frequency_enhance(object):
    def __init__(self, low_freq_threshold=30, high_freq_boost=1.5):
        """
        初始化DCT分割器。

        参数:
        - low_freq_threshold: 低频部分的阈值，默认为 30
        - high_freq_boost: 高频部分的增强系数，默认$beta$为 1.5
        """
        self.low_freq_threshold = low_freq_threshold
        self.beta = high_freq_boost

    def __call__(self, x):
        """
        对输入图像进行DCT分割，提取低频部分和高频部分。

        参数:
        - x: 输入图像张量，形状为 [batch_size, channels, height, width]

        返回:
        - 低频部分和高频部分的张量，形状相同于输入图像
        """
        # 对输入图像进行DCT变换
        x_dct = dct_2d(x)
        # 生成低频掩码
        low_freq_mask = torch.zeros_like(x_dct)
        low_freq_mask[:, :self.low_freq_threshold, :self.low_freq_threshold] = 1

        # 提取低频和高频部分
        low_freq_dct = x_dct * low_freq_mask
        high_freq_dct = x_dct - low_freq_dct

        # 计算逆DCT得到低频和高频图像
        x_low_freq = idct_2d(low_freq_dct)
        x_high_freq = idct_2d(high_freq_dct * self.beta)

        return x_low_freq+x_high_freq


class ImageSplitterAndShuffler(object):
    def __init__(self):
        pass

    def __call__(self, images):
        """
        shuffle
        """
        channels, height, width = images.size()

        images_ = torch.zeros_like(images)

        block_height = height // 2
        block_width = width // 2

        block1 = images[:, :block_height, :block_width]
        block2 = images[:, :block_height, block_width:]
        block3 = images[:, block_height:, :block_width]
        block4 = images[:, block_height:, block_width:]

        blocks = [block1, block2, block3, block4]
        random.shuffle(blocks)

        images_[:, :block_height, :block_width] = blocks[0]
        images_[:, :block_height, block_width:] = blocks[1]
        images_[:, block_height:, :block_width] = blocks[2]
        images_[:, block_height:, block_width:] = blocks[3]

        return images_

class Normalize(nn.Module):
    def __init__(self, ms=None):
        super(Normalize, self).__init__()
        if ms == None:
            self.ms = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]

    def forward(self, input):
        x = input.clone()
        for i in range(x.shape[1]):
            x[:, i] = (x[:, i] - self.ms[0][i]) / self.ms[1][i]
        return x

class CUB200_PIL(Dataset):
    def __init__(self, data_path, img_filepath):
        self.img_path = data_path
        # reading img file from file
        fp = open(img_filepath, 'rt')
        self.img_filename = []
        self.label = []
        for x in fp:
            x = x.strip()
            self.img_filename.append(x.split(' ')[0])
            self.label.append(int(x.strip().split(' ')[1]))
        fp.close()

    def __getitem__(self, index):
        with open(os.path.join(self.img_path, self.img_filename[index]), 'rb') as f:
            bin_data = f.read()
        label = self.label[index]
        return bin_data, label

    def __len__(self):
        return len(self.img_filename)

class CUB200_PIL_nolabel(Dataset):

    def __init__(self, data_path, img_filepath, transform=None):
        self.img_path = data_path
        self.transform = transform
        # reading img file from file
        fp = open(img_filepath, 'rt')
        self.img_filename = []
        for x in fp:
            x = x.strip()
            self.img_filename.append(x)
        fp.close()


    def __getitem__(self, index):
        f = open(os.path.join(self.img_path, self.img_filename[index]), 'rb')
        img = Image.open(f)
        img = img.convert('RGB')
        imgsize = img.size
        if self.transform is not None:
            img = self.transform(img)
        return img, (imgsize[1], imgsize[0])#, label#, self.img_filename[index]

    def __len__(self):
        return len(self.img_filename)


class ElementwisePair(CUB200_PIL_nolabel):

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        pos_1 = self.transform(img)
        pos_2 = self.transform(img)

        return pos_1, pos_2, self.img_filename[index]

class ImageFolderPair(torchvision.datasets.folder.ImageFolder):
    """Pair Dataset.
    """

    def __getitem__(self, index):
        img = Image.open(self.imgs[index][0])
        if img.mode != 'RGB':
            img = img.convert('RGB')

        pos_1 = self.transform(img)
        pos_2 = self.transform(img)

        return pos_1, pos_2, self.imgs[index][1]


class ImageFolderTriple(torchvision.datasets.folder.ImageFolder):
    """triplet Dataset.
    """
    def __init__(self, root, transform, trans_ori):
        super().__init__(root, transform)
        self.trans_ori = trans_ori

    def __getitem__(self, index):
        img = Image.open(self.imgs[index][0])
        if img.mode != 'RGB':
            img = img.convert('RGB')

        pos_1 = self.transform(img)
        pos_2 = self.transform(img)
        pos = self.trans_ori(img)

        return pos_1, pos_2, pos, self.imgs[index][1]


class ElementwiseTriple(CUB200_PIL_nolabel):

    def __init__(self, data_path, img_filepath, transform, trans_ori):
        super().__init__(data_path, img_filepath, transform)
        self.trans_ori = trans_ori

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        pos_1 = self.transform(img)
        pos_2 = self.transform(img)
        pos = self.trans_ori(img)
        return pos_1, pos_2, pos, self.img_filename[index]


class ImageFolderLMDB(Dataset):
    def __init__(self, db_path, trans_1=None,trans_2=None,trans_ori=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = self.loads_data(txn.get(b'__len__'))
            self.keys = self.loads_data(txn.get(b'__keys__'))

        self.transform_1 = trans_1
        self.transform_2 = trans_2
        self.trans_ori = trans_ori

    def loads_data(self, buf):
        """
        Args:
            buf: the output of `dumps`.
        """
        return pickle.loads(buf)

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = self.loads_data(byteflow)

        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        # target = unpacked[1]

        if self.transform_1 and self.transform_2 is not None:
            pos_1 = self.transform_1(img)
            pos_2 = self.transform_2(img)
        pos = self.trans_ori(img)

        return pos_1, pos_2, pos


    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


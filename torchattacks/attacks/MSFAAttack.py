import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import *


import numpy as np
from ..attack import Attack

mid_outputs = []
class MSFAAttack(Attack):
    r"""
    Multi-Scale Feature Alignment Attack
    """
    def __init__(self, model, pretrained, eps=8/255, alpha=1/255, steps=8, mu=0.9, targeted=False):
        super(MSFAAttack, self).__init__("MSFAAttack", model)
        self.eps = eps
        self.steps = steps
        self.mu = mu
        self.tar = targeted
        if self.tar:
            self.alpha = -alpha
        else:
            self.alpha = alpha
        self.use_pretrained = pretrained

    def forward(self, images):
        r"""
        Overridden.
        """
        images = images.clone().detach().cuda()
        momentum = torch.zeros_like(images).detach().cuda()
        adv_images = images.clone().detach() + torch.empty_like(images).uniform_(-self.eps, self.eps)
        adv_images = random_transform(adv_images)

        if not self.use_pretrained:
            feature_layers = ['4', '5', '6']
        else:
            feature_layers = ['layer1', 'layer2', 'layer3']

        # mid_outputs = []
        global mid_outputs

        def get_mid_output(m, i, o):
            global mid_outputs
            mid_outputs.append(o)

        hs = []
        for layer_name in feature_layers:
            if not self.use_pretrained:
                hs.append(self.model.f._modules.get(layer_name).register_forward_hook(get_mid_output))
            else:
                hs.append(self.model[1]._modules.get(layer_name).register_forward_hook(get_mid_output))

        _, out_1 = self.model(images)

        mid_originals = []
        for mid_output in mid_outputs:
            mid_original = torch.zeros(mid_output.size()).cuda()
            mid_originals.append(mid_original.copy_(mid_output))
        mid_outputs = []

        for i in range(self.steps):
            adv_images.requires_grad = True
            _, out_2 = self.model(adv_images)

            mid_originals_ = []
            for mid_original in mid_originals:
                mid_originals_.append(mid_original.detach())

            n_img = mid_originals_[0].shape[0]

            loss_mid1 = 1 - F.cosine_similarity(mid_originals_[0].reshape(n_img, -1),
                                                mid_outputs[0].reshape(n_img, -1)).mean()
            loss_mid2 = 1 - F.cosine_similarity(mid_originals_[1].reshape(n_img, -1),
                                                mid_outputs[1].reshape(n_img, -1)).mean()
            loss_mid3 = 1 - F.cosine_similarity(mid_originals_[2].reshape(n_img, -1),
                                                mid_outputs[2].reshape(n_img, -1)).mean()
            loss_out = 1 - F.cosine_similarity(out_1, out_2).mean()

            loss_final = loss_mid1 + loss_mid2 + loss_mid3 + loss_out
            cost = loss_final.cuda()

            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            momentum = momentum * self.mu + grad

            adv_images = adv_images.detach() + self.alpha * momentum.sign()

            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            mid_outputs = []
        print(cost)
        for h in hs:
            h.remove()

        return adv_images

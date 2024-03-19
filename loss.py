import torch
from torch import nn
from torchvision.models.vgg import vgg16
import torch.nn.functional as F

import torch.nn as nn
from torchvision.models.vgg import vgg16, vgg19
import cv2
import numpy as np
#
# class PerceptualLoss(nn.Module):
#     def __init__(self):
#         super(PerceptualLoss, self).__init__()
#         vgg = vgg16(pretrained=True).cuda()
#         self.loss_network = nn.Sequential(*list(vgg.features)[:16]).eval()
#         for param in self.loss_network.parameters():
#             param.requires_grad = False
#         self.mse_loss = nn.MSELoss()
#         self.l1_loss = nn.L1Loss()
#
#     def forward(self, out_images, target_images):
#         loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
#         return loss



# --- Perceptual loss network  --- #
class PerceptualLoss(nn.Module):

    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(pretrained=True).cuda()
        self.loss_network = nn.Sequential(*list(vgg.features)[:16]).eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False

        self.l1_loss = nn.L1Loss()

    def normalize_batch(self, batch):
        # Normalize batch using ImageNet mean and std
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        return (batch - mean) / std

    def forward(self, out_images, target_images):

        loss = self.l1_loss(
            self.loss_network(self.normalize_batch(out_images)),
            self.loss_network(self.normalize_batch(target_images))
        )

        return loss



class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        # loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        # loss_network = nn.Sequential(*list(vgg.features)[:30]).eval()
        # loss_network36 = nn.Sequential(*list(vgg.features)[:36]).cuda().eval()
        loss_network35 = nn.Sequential(*list(vgg.features)[:35]).cuda().eval()

        # for param in loss_network35.parameters():
        #     param.requires_grad = False


        for param in loss_network35.parameters():
            param.requires_grad = False
        # self.loss_network36 = loss_network36
        self.loss_network35 = loss_network35

        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, target_images, out_images):
        # out_images = out_images.cuda()
        # Adversarial Loss
        # adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        # perception_loss = F.l1_loss(self.loss_network35(out_images), self.loss_network35(target_images))
        # perception_loss = F.l1_loss(self.loss_network35(out_images), self.loss_network35(target_images))

        # import scipy.io
        # for i in range(0,512):
        #     scipy.io.savemat('./log_36/temp36_%04d.mat' % (i), {'x': torch.log(abs(self.loss_network36(out_images))+1)[0, i, ...].unsqueeze(2).cpu().numpy()})
        #     scipy.io.savemat('./log_35/temp35_%04d.mat' % (i), {'x': torch.log(abs(self.loss_network35(out_images))+1)[0, i, ...].unsqueeze(2).cpu().numpy()})
            # scipy.io.savemat('./27/temp36_%04d.mat' % (i), {'x': self.loss_network36(out_images)[0, i, ...].unsqueeze(2).numpy()})
            # scipy.io.savemat('./26/temp35_%04d.mat' % (i), {'x': self.loss_network35(out_images)[0, i, ...].unsqueeze(2).numpy()})

        perception_loss = torch.exp(F.l1_loss(self.relu(torch.log(torch.abs(self.loss_network35(out_images))+1)), self.relu(torch.log(torch.abs(self.loss_network35(target_images))+1)))) - 1
        # vgg_out = self.loss_network(out_images)

        # Image Loss
        # image_loss = F.l1_loss(out_images, target_images)
        image_loss = self.mse_loss(out_images, target_images)

        # # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss +  2e-8 * tv_loss + 0.006 * perception_loss

    # return image_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

from torch.autograd import Variable
if __name__ == "__main__":
    img = cv2.imread('baboon.png').astype(np.float32) / 255.
    img = Variable(torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0))
    g_loss = GeneratorLoss()
    tempq = g_loss(img)
    temp = 1
    # cv2.imwrite('vgg_out.png', _loss)
    # print(g_loss)
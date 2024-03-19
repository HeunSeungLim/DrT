import os
import random

import cv2
import numpy as np
import torch
import h5py
import time

import torch.nn as nn
from torch.nn import init
import torchvision as tv
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

def mk_trained_dir_if_not(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


from glob import glob
def model_restore(model, trained_model_dir):
    model_list = glob((trained_model_dir + "/*.pkl"))
    a = []
    for i in range(len(model_list)):
        index = int(model_list[i].split('model')[-1].split('.')[0])
        a.append(index)
    epoch = np.sort(a)[-1]
    # model_path = os.path.join(trained_model_dir, '/PSNR_7.55_trained_model79.pkl')
    model_path = trained_model_dir + 'trained_model{}.pkl'.format(epoch)
    model.load_state_dict(torch.load(model_path))
    return model, epoch


class data_loader(data.Dataset):
    def __init__(self, list_dir):
        f = open(list_dir)
        self.list_txt = f.readlines()
        self.length = len(self.list_txt)



    def __getitem__(self, index):

        sample_path = self.list_txt[index][:-1]

        if os.path.exists(sample_path):

            f = h5py.File(sample_path, 'r')
            data = f['IN'][:]
            label = f['GT'][:]
            f.close()
            crop_size = 256
            data, label = self.imageCrop(data, label, crop_size)
            data, label = self.image_Geometry_Aug(data, label)

        # print(sample_path)
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return self.length

    def random_number(self, num):
        return random.randint(1, num)

    def imageCrop(self, data, label, crop_size):
        c, w, h = data.shape
        w_boder = w - crop_size  # sample point y
        h_boder = h - crop_size  # sample point x ...

        start_w = self.random_number(w_boder - 1)
        start_h = self.random_number(h_boder - 1)

        crop_data = data[:, start_w:start_w + crop_size, start_h:start_h + crop_size]
        crop_label = label[:, start_w:start_w + crop_size, start_h:start_h + crop_size]
        return crop_data, crop_label

    def image_Geometry_Aug(self, data, label):
        c, w, h = data.shape
        num = self.random_number(4)

        if num == 1:
            in_data = data
            in_label = label

        if num == 2:  # flip_left_right
            index = np.arange(w, 0, -1) - 1
            in_data = data[:, index, :]
            in_label = label[:, index, :]

        if num == 3:  # flip_up_down
            index = np.arange(h, 0, -1) - 1
            in_data = data[:, :, index]
            in_label = label[:, :, index]

        if num == 4:  # rotate 180
            index = np.arange(w, 0, -1) - 1
            in_data = data[:, index, :]
            in_label = label[:, index, :]
            index = np.arange(h, 0, -1) - 1
            in_data = in_data[:, :, index]
            in_label = in_label[:, :, index]

        return in_data, in_label

def get_lr(epoch, lr, max_epochs):
    if epoch <= max_epochs * 0.8:
        lr = lr
    else:
        lr = 0.1 * lr
    return lr



from loss import GeneratorLoss



def rgb2hsv(input, epsilon=1e-10):
    assert(input.shape[1] == 3)

    r, g, b = input[:, 0], input[:, 1], input[:, 2]
    max_rgb, argmax_rgb = input.max(1)
    min_rgb, argmin_rgb = input.min(1)

    max_min = max_rgb - min_rgb + epsilon

    h1 = 60.0 * (g - r) / max_min + 60.0
    h2 = 60.0 * (b - g) / max_min + 180.0
    h3 = 60.0 * (r - b) / max_min + 300.0

    h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
    s = max_min / (max_rgb + epsilon)
    v = max_rgb

    return h, s, v

def trainqqq(epoch, model, netD, train_loaders, optimizer, optimizer_D, args):
    lr = get_lr(epoch, args.lr, args.epochs)
    lossed = GeneratorLoss().cuda()
    # HED = Network
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('lr: {}'.format(optimizer.param_groups[0]['lr']))
    model.train()
    netD.train()
    num = 0
    trainloss = 0
    start = time.time()

    criterion_GAN = torch.nn.BCEWithLogitsLoss().cuda()
    criterion_content = torch.nn.L1Loss().cuda()
    criterion_pixel = torch.nn.L1Loss().cuda()
    from tqdm import tqdm
    # train_bar = tqdm(train_loaders)

    for batch_idx, (data, target) in enumerate(train_loaders):
        if args.use_cuda:
            data, target = data.cuda(), target.cuda()
        end = time.time()

############  used for End-to-End code
        data1 = torch.cat((data[:, 0:3, :, :], data[:, 9:12, :, :]), dim=1)
        data2 = torch.cat((data[:, 3:6, :, :], data[:, 12:15, :, :]), dim=1)
        data3 = torch.cat((data[:, 6:9, :, :], data[:, 15:18, :, :]), dim=1)
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        data1 = Variable(data1)
        data2 = Variable(data2)
        data3 = Variable(data3)
        target = Variable(target)
        optimizer.zero_grad()
        output = model(data1, data2, data3)

#########  make the loss
        output = torch.log(1 + 5000 * output.cpu()) / torch.log(
            Variable(torch.from_numpy(np.array([1 + 5000])).float()))
        target = torch.log(1 + 5000 * target).cpu() / torch.log(
            Variable(torch.from_numpy(np.array([1 + 5000])).float()))
        temp = hsv2rgb(output)
        loss = F.l1_loss(output, target)
        loss.backward()
        optimizer.step()

        netD.zero_grad()
        output = model(data1, data2, data3)
        real_out = netD(target).mean()
        fake_out = netD(output).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)
        optimizer_D.step()


        trainloss = trainloss + loss
        if (batch_idx + 1) % 4 == 0:
            trainloss = trainloss / 4
            print('train Epoch {} iteration: {} loss: {:.6f}'.format(epoch, batch_idx, trainloss.data))
            fname = args.trained_model_dir + 'lossTXT.txt'
            try:
                fobj = open(fname, 'a')

            except IOError:
                print('open error')
            else:
                fobj.write('train Epoch {} iteration: {} Loss: {:.6f}\n'.format(epoch, batch_idx, trainloss.data))
                fobj.close()
            trainloss = 0





def train(epoch, model, train_loaders, optimizer, args):
    lr = get_lr(epoch, args.lr, args.epochs)
    lossed = GeneratorLoss().cuda()
    # HED = Network
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('lr: {}'.format(optimizer.param_groups[0]['lr']))
    model.train()
    num = 0
    trainloss = 0
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loaders):
        if args.use_cuda:
            data, target = data.cuda(), target.cuda()
        end = time.time()

############  used for End-to-End code
        data1 = torch.cat((data[:, 0:3, :, :], data[:, 9:12, :, :]), dim=1)
        data2 = torch.cat((data[:, 3:6, :, :], data[:, 12:15, :, :]), dim=1)
        data3 = torch.cat((data[:, 6:9, :, :], data[:, 15:18, :, :]), dim=1)

        data1 = Variable(data1)
        data2 = Variable(data2)
        data3 = Variable(data3)
        target = Variable(target)
        optimizer.zero_grad()
        # output = model(Variable(data))
        final = model(data1, data2, data3)

#########  make the loss
        # from pytorch_hsv import HSVLoss
        # rgb2hsv_get = HSVLoss()
        # output_corse = torch.log(1 + 5000 * output_corse.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
        target = torch.log(1 + 5000 * target).cpu() / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
        final = torch.log(1 + 5000 * final.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
        # output_h, output_s, output_v = rgb2hsv_get.get_hsv(final)
        # real_h, real_s, real_v = rgb2hsv_get.get_hsv(target)

        # color_loss = (F.l1_loss(output_h, real_h) + F.l1_loss(output_s, real_s)) * 0.01

        loss = lossed(target.cuda(), final.cuda())
        loss.backward()
        optimizer.step()
        trainloss = trainloss + loss
        if (batch_idx +1) % 4 == 0:
            trainloss = trainloss / 4
            print('train Epoch {} iteration: {} loss: {:.6f}'.format(epoch, batch_idx, trainloss.data))
            fname = args.trained_model_dir + 'lossTXT.txt'
            try:
                fobj = open(fname, 'a')

            except IOError:
                print('open error')
            else:
                fobj.write('train Epoch {} iteration: {} Loss: {:.6f}\n'.format(epoch, batch_idx, trainloss.data))
                fobj.close()
            trainloss = 0
from torch.autograd import Variable

from math import log10, log
def psnr(x, target):
    sqrdErr = torch.mean((x - target) ** 2)
    return 10 * log10(1/sqrdErr)
def range_compressor(x):
    return (torch.log(x.mul(5000).add(1))) / log(1 + 5000)
import imageio


def get_patch(img,size=500):
    h,w = img.size(2)//size , img.size(3)//size
    x1 = torch.split(img,size,2)
    patch = []
    for i in range(h):
        for j in range(w):
            patch.append(torch.split(x1[i],size,3)[j])

    return patch

def testing_fun(model, test_loaders, args):
    model.eval()
    total_psnr_linear = 0
    total_psnr_mu = 0
    num = 0
    for data, target in test_loaders:
        # Test_Data_name = test_loaders.dataset.img_list[num].split('.h5')[0].split('/')[-1]
        with torch.no_grad():

            data = data.transpose(2, 3)
            target = target.transpose(2, 3)

            input_patch = get_patch(data,256)
            target_patch = get_patch(target,256)
            psnr_linear, psnr_mu = 0, 0
            for i in range(len(input_patch)):
                data1 = torch.cat((input_patch[i][:,0:3,:,:],input_patch[i][:,9:12,:,:]),1)
                data2 = torch.cat((input_patch[i][:, 3:6, :, :], input_patch[i][:, 12:15, :, :]), 1)
                data3 = torch.cat((input_patch[i][:, 6:9, :, :], input_patch[i][:, 15:18, :, :]), 1)
                output_patch = model(data1.cuda(), data2.cuda(), data3.cuda())
                output_patch = output_patch.cpu()
                psnr_linear += psnr(output_patch, target_patch[i])
                psnr_mu += psnr(range_compressor(output_patch),range_compressor(target_patch[i]))
            psnr_linear /= len(input_patch)
            psnr_mu /= len(input_patch)

########################################
            # data, target = data.cuda(), target.cuda()



        # data1 = torch.cat((data[:, 0:3, :], data[:, 9:12, :]), dim=1)
        # data2 = torch.cat((data[:, 3:6, :], data[:, 12:15, :]), dim=1)
        # data3 = torch.cat((data[:, 6:9, :], data[:, 15:18, :]), dim=1)
        # data1 = Variable(data1, volatile=True)
        # data2 = Variable(data2, volatile=True)
        # data3 = Variable(data3, volatile=True)
        # target = Variable(target, volatile=True)
        # output = model(data1, data2, data3)

        # val = psnr(output,target)
        # mu_val = psnr(range_compressor(output),range_compressor(target))
        # save the result to .H5 files

        # output = torch.squeeze(output)
        # target = torch.squeeze(target)
        # output = output.cpu()
        # target = target.cpu()
        # output = output.detach().numpy()
        # target = target.detach().numpy()
        # output = np.rollaxis(output, 0, start=3)
        # target = np.rollaxis(output, 0, start=3)
        # imageio.imsave('./result/' + Test_Data_name + '_hdr.hdr', output, format='hdr')


        # hdrfile = h5py.File(args.result_dir + Test_Data_name + '_hdr.h5', 'w')
        # img = output[0, :, :, :]
        # img = tv.utils.make_grid(img.data.cpu()).numpy()
        # hdrfile.create_dataset('data', data=img)
        # hdrfile.close()

        # hdr = torch.log(1 + 5000 * output.cpu()) / torch.log(
        #     Variable(torch.from_numpy(np.array([1 + 5000])).float()))
        # target = torch.log(1 + 5000 * target).cpu() / torch.log(
        #     Variable(torch.from_numpy(np.array([1 + 5000])).float()))

        # test_loss += F.mse_loss(hdr, target)
        # num = num + 1
        # psnr = cv2.PSNR(output,target)
        # mu = cv2.PSNR(log((1 + 5000*output))/log(1+5000),log((1 + 5000*target))/log(1+5000))
        total_psnr_linear += psnr_linear
        total_psnr_mu += psnr_mu
    avg_psnr_linear = total_psnr_linear / len(test_loaders.dataset)
    avg_psnr_mu = total_psnr_mu/ len(test_loaders.dataset)
    print('\n l-psnr:' + str(avg_psnr_linear))
    print('\n mu-psnr:' + str(avg_psnr_mu))
    return avg_psnr_mu

def testing_fun_(model, test_loaders, args):
    model.eval()
    test_loss = 0
    mu_psnr = 0
    num = 0

    l_ssim = 0
    mu_ssim = 0


    for data, target in test_loaders:
        Test_Data_name = test_loaders.dataset.list_txt[num].split('.h5')[0].split('/')[-1]
        with torch.no_grad():


########################################

            data = data.transpose(2,3)
            target = target.transpose(2,3)
            padded = torch.zeros([1,18,1024,1536])
            padded[:,:,12:1012,18:1518] = data
            data = padded
            padded = torch.zeros([1,3,1024,1536])
            padded[:,:,12:1012,18:1518] = target
            target = padded
            patch_size = 256
            x_num = data.size(2) // patch_size
            y_num = data.size(3) // patch_size

            # output = model(data1, data2, data3)

            output = torch.zeros([1,3,data.size(2),data.size(3)])
            for i in range(x_num):
                for j in range(y_num):
                    patch_input = data[:,:,patch_size * i : patch_size * (i + 1), patch_size * j : patch_size * (j + 1)]

                    data1 = torch.cat((patch_input[:, 0:3, :], patch_input[:, 9:12, :]), dim=1)
                    data2 = torch.cat((patch_input[:, 3:6, :], patch_input[:, 12:15, :]), dim=1)
                    data3 = torch.cat((patch_input[:, 6:9, :], patch_input[:, 15:18, :]), dim=1)

                    output[:,:,patch_size * i : patch_size * (i + 1), patch_size * j : patch_size * (j + 1)] = model(data1.cuda(), data2.cuda(), data3.cuda())


########################################
            # data, target = data.cuda(), target.cuda()



        # data1 = torch.cat((data[:, 0:3, :], data[:, 9:12, :]), dim=1)
        # data2 = torch.cat((data[:, 3:6, :], data[:, 12:15, :]), dim=1)
        # data3 = torch.cat((data[:, 6:9, :], data[:, 15:18, :]), dim=1)
        # data1 = Variable(data1, volatile=True)
        # data2 = Variable(data2, volatile=True)
        # data3 = Variable(data3, volatile=True)
        # target = Variable(target, volatile=True)
        # output = model(data1, data2, data3)

        val = psnr(output,target)
        mu_val = psnr(range_compressor(output),range_compressor(target))



        # save the result to .H5 files

        output = torch.squeeze(output)
        # target = torch.squeeze(target)
        output = output.cpu()
        # target = target.cpu()
        output = output.detach().numpy()
        # target = target.detach().numpy()
        output = np.rollaxis(output, 0, start=3)
        # target = np.rollaxis(output, 0, start=3)
        imageio.imsave('./result/' + Test_Data_name + '_hdr.hdr', output, format='hdr')
        # imageio.imsave('./result/' + Test_Data_name + '_hdr.hdr', output, format='hdr')


        # hdrfile = h5py.File(args.result_dir + Test_Data_name + '_hdr.h5', 'w')
        # img = output[0, :, :, :]
        # img = tv.utils.make_grid(img.data.cpu()).numpy()
        # hdrfile.create_dataset('data', data=img)
        # hdrfile.close()

        # hdr = torch.log(1 + 5000 * output.cpu()) / torch.log(
        #     Variable(torch.from_numpy(np.array([1 + 5000])).float()))
        # target = torch.log(1 + 5000 * target).cpu() / torch.log(
        #     Variable(torch.from_numpy(np.array([1 + 5000])).float()))

        # test_loss += F.mse_loss(hdr, target)
        num = num + 1
        # psnr = cv2.PSNR(output,target)
        test_loss += val
        mu_psnr += mu_val

        # l_ssim += ssim_score
        # mu_ssim += mu_ssim_score

    test_loss = test_loss / len(test_loaders.dataset)
    mu_psnr = mu_psnr/ len(test_loaders.dataset)
    print('\n l-psnr:' + str(test_loss))
    print('\n mu-psnr:' + str(mu_psnr))

    # l_ssim = l_ssim / len(test_loaders.dataset)
    # mu_ssim = mu_ssim / len(test_loaders.dataset)
    # print('\n l-ssim' + str(ssim))
    # print('\n mu-ssim:' + str(mu_ssim))

    return mu_psnr

def testing_fun_1(model, test_loaders, args):
    model.eval()

    test_loss = 0
    num = 0
    mu_psnr = 0

    l_ssim = 0
    mu_ssim= 0

    for data, target in test_loaders:
        with torch.no_grad():
            Test_Data_name = test_loaders.dataset.list_txt[num].split('.h5')[0].split('/')[-1]
            if args.use_cuda:
                data, target = data.cuda(), target.cuda()
            data = data.transpose(2, 3)
            target = target.transpose(2, 3)
            data1 = torch.cat((data[:, 0:3, :], data[:, 9:12, :]), dim=1)
            data2 = torch.cat((data[:, 3:6, :], data[:, 12:15, :]), dim=1)
            data3 = torch.cat((data[:, 6:9, :], data[:, 15:18, :]), dim=1)

            b, c, row, col = data1.size()

            cut_size = 1000
            pad_size = 100

            index_row = np.arange(0, row - cut_size, cut_size)
            index_row = np.append(index_row, row - cut_size)
            index_col = np.arange(0, col - cut_size, cut_size)
            index_col = np.append(index_col, col - cut_size)
            m = torch.nn.ReplicationPad2d((pad_size, pad_size, pad_size, pad_size))

            data1 = m(data1)
            data2 = m(data2)
            data3 = m(data3)

            prediction = torch.zeros(b, 3, row * 1, col * 1)
            prediction = Variable(prediction)
            # att1 = torch.zeros(b, 64, row * 1, col * 1)
            # att1 = Variable(att1)
            # att2 = torch.zeros(b, 64, row * 1, col * 1)
            # att2 = Variable(att2)

            for i in index_row:
                for j in index_col:
                    cut_input1 = data1[:, :, i:i + pad_size * 2 + cut_size, j:j + pad_size * 2 + cut_size].cuda()
                    cut_input2 = data2[:, :, i:i + pad_size * 2 + cut_size, j:j + pad_size * 2 + cut_size].cuda()
                    cut_input3 = data3[:, :, i:i + pad_size * 2 + cut_size, j:j + pad_size * 2 + cut_size].cuda()
                    # cut_inputv1 = inputv1[:, :, i:i + pad_size * 2 + cut_size, j:j + pad_size * 2 + cut_size]


                    # cut_output , _1, _2 = model(cut_input1, cut_input2, cut_input3)
                    cut_output = model(cut_input1, cut_input2, cut_input3)

                    prediction[:, :, i:i + cut_size, j:j + cut_size] = cut_output.data[:, :, pad_size:pad_size + cut_size, pad_size:pad_size + cut_size]
                    # att1[:, :, i:i + cut_size, j:j + cut_size] = _1.data[:, :, pad_size:pad_size + cut_size,
                    #                                                    pad_size:pad_size + cut_size]
                    # att2[:, :, i:i + cut_size, j:j + cut_size] = _2.data[:, :, pad_size:pad_size + cut_size,
                    #                                                    pad_size:pad_size + cut_size]
            #
            # prediction = prediction[..., :h, :w]
            # att1 = np.mean(att1[0, ...].detach().numpy().transpose(1, 2, 0), 2)
            # att2 = np.mean(att2[0, ...].detach().numpy().transpose(1, 2, 0), 2)
            #
            # cv2.imwrite('att1.png', np.uint8(np.clip(att1, 0, 1) * 255))
            # cv2.imwrite('att2.png', np.uint8(np.clip(att2, 0, 1) * 255))

            output = prediction.cuda()
            # save the result to .H5 files
            val = psnr(output, target)
            mu_val = psnr(range_compressor(output), range_compressor(target))

            from IQA_pytorch import SSIM, utils
            from PIL import Image
            # import torch

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # ref_path = 'r0.png'
            # dist_path = 'r1.png'

            # ref = utils.prepare_image(Image.open(ref_path).convert("RGB")).to(device)
            # dist = utils.prepare_image(Image.open(dist_path).convert("RGB")).to(device)
            # with torch.no_grad():
            model_ssim = SSIM(channels=3).cuda()
            val_ssim_val = model_ssim(output, target, as_loss=False)
            mu_ssim_val = model_ssim(range_compressor(output), range_compressor(target), as_loss=False)
            # print('score: %.4f' % score.item())

            # val_ssim_val = ssim(torch.mean(output[0, ...],0).cpu().detach().numpy()[np.newaxis, ...].transpose(1,2,0),
            #                     torch.mean(target[0, ...],0).cpu().detach().numpy()[np.newaxis, ...].transpose(1,2,0))
            # mu_ssim_val = ssim(range_compressor(output), range_compressor(target))

            # save the result to .H5 files
            temp_out = range_compressor(output)
            temp_tar = range_compressor(target)

            output = torch.squeeze(output)
            # target = torch.squeeze(target)
            output = output.cpu()
            # target = target.cpu()
            output = output.detach().numpy()

            cv2.imwrite('output1.png',
                        cv2.cvtColor(np.uint8(np.clip(temp_out[0,...].cpu().detach().numpy().transpose(1, 2, 0), 0, 1) * 255), cv2.COLOR_RGB2BGR))
            cv2.imwrite('target.png',
                        cv2.cvtColor(np.uint8(np.clip(temp_tar[0,...].cpu().detach().numpy().transpose(1, 2, 0), 0, 1) * 255), cv2.COLOR_RGB2BGR))

            # target = target.detach().numpy()
            # output = np.rollaxis(output, 0, start=3)
            # target = np.rollaxis(output, 0, start=3)
            # imageio.imsave('./result/' + Test_Data_name + '_hdr.hdr', output, format='hdr')

            # hdrfile = h5py.File(args.result_dir + Test_Data_name + '_hdr.h5', 'w')
            # img = output
            # img = tv.utils.make_grid(img.data.cpu()).numpy()
            # hdrfile.create_dataset('data', data=img)
            # hdrfile.close()
            output = np.rollaxis(output, 0, start=3)
            # cv2.cvtColor()
            cv2.imwrite(args.result_dir + Test_Data_name + '_hdr.hdr', cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            # imageio.imsave(args.result_dir + Test_Data_name + '_hdr.hdr', output, format='hdr')
            # hdr = torch.log(1 + 5000 * output.cpu()) / torch.log(
            #     Variable(torch.from_numpy(np.array([1 + 5000])).float()))
            # target = torch.log(1 + 5000 * target).cpu() / torch.log(
            #     Variable(torch.from_numpy(np.array([1 + 5000])).float()))

            # test_loss += F.mse_loss(hdr, target)
            num = num + 1
            # psnr = cv2.PSNR(output,target)
            test_loss += val
            mu_psnr += mu_val
            l_ssim += val_ssim_val
            mu_ssim += mu_ssim_val
            # l_ssim += ssim_score
            # mu_ssim += mu_ssim_score

    test_loss = test_loss / len(test_loaders.dataset)
    mu_psnr = mu_psnr / len(test_loaders.dataset)
    l_ssim = l_ssim / len(test_loaders.dataset)
    mu_ssim = mu_ssim / len(test_loaders.dataset)

    print('\n l-psnr:' + str(test_loss))
    print('\n mu-psnr:' + str(mu_psnr))
    print('\n l-ssim:' + str(l_ssim))
    print('\n mu-ssim:' + str(mu_ssim))

    # test_loss += F.mse_loss(hdr, target)
        # num = num + 1

    # test_loss = test_loss / len(test_loaders.dataset)
    # print('\n Test set: Average Loss: {:.4f}'.format(test_loss.data[0]))

    return mu_psnr


def test_split(model, L1, L2, L3, refield=32, min_size=256, sf=1, modulo=1):
    E = test_split_fn(model, L1, L2, L3, refield=refield, min_size=min_size, sf=sf, modulo=modulo)
    return E

def test_split_fn(model, L1, L2, L3, refield=32, min_size=256, sf=1, modulo=1):
    '''
    model:
    L: input Low-quality image
    refield: effective receptive filed of the network, 32 is enough
    min_size: min_sizeXmin_size image, e.g., 256X256 image
    sf: scale factor for super-resolution, otherwise 1
    modulo: 1 if split
    '''
    h, w = L1.size()[-2:]
    if h*w <= min_size**2:
        L1 = torch.nn.ReplicationPad2d((0, int(np.ceil(w/modulo)*modulo-w), 0, int(np.ceil(h/modulo)*modulo-h)))(L1)
        L2 = torch.nn.ReplicationPad2d(
            (0, int(np.ceil(w / modulo) * modulo - w), 0, int(np.ceil(h / modulo) * modulo - h)))(L2)
        L3 = torch.nn.ReplicationPad2d(
            (0, int(np.ceil(w / modulo) * modulo - w), 0, int(np.ceil(h / modulo) * modulo - h)))(L3)

        E = model(L1, L2, L3)
        E = E[..., :h*sf, :w*sf]
    else:
        top = slice(0, (h//2//refield+1)*refield)
        bottom = slice(h - (h//2//refield+1)*refield, h)
        left = slice(0, (w//2//refield+1)*refield)
        right = slice(w - (w//2//refield+1)*refield, w)
        Ls1 = [L1[..., top, left], L1[..., top, right], L1[..., bottom, left], L1[..., bottom, right]]
        Ls2 = [L2[..., top, left], L2[..., top, right], L2[..., bottom, left], L2[..., bottom, right]]
        Ls3 = [L3[..., top, left], L3[..., top, right], L3[..., bottom, left], L3[..., bottom, right]]


        if h * w <= 4*(min_size**2):
            Es = [model(Ls1[i], Ls2[i], Ls3[i]) for i in range(4)]
        else:
            Es = [test_split_fn(model, Ls1[i], Ls2[i], Ls3[i], refield=refield, min_size=min_size, sf=sf, modulo=modulo) for i in range(4)]

        b, c = Es[0].size()[:2]
        E = torch.zeros(b, c, sf * h, sf * w).type_as(L1)


        E[..., :h//2*sf, :w//2*sf] = Es[0][..., :h//2*sf, :w//2*sf]
        E[..., :h//2*sf, w//2*sf:w*sf] = Es[1][..., :h//2*sf, (-w + w//2)*sf:]
        E[..., h//2*sf:h*sf, :w//2*sf] = Es[2][..., (-h + h//2)*sf:, :w//2*sf]
        E[..., h//2*sf:h*sf, w//2*sf:w*sf] = Es[3][..., (-h + h//2)*sf:, (-w + w//2)*sf:]
    return E

def testing_fun_1_(model, test_loaders):
    model.eval()
    test_loss = 0
    num = 0
    mu_psnr = 0
    for data, target in test_loaders:
        with torch.no_grad():
            Test_Data_name = test_loaders.dataset.list_txt[num].split('.h5')[0].split('/')[-1]
            # if args.use_cuda:
            data, target = data.cuda(), target.cuda()

            data1 = torch.cat((data[:, 0:3, :], data[:, 9:12, :]), dim=1)
            data2 = torch.cat((data[:, 3:6, :], data[:, 12:15, :]), dim=1)
            data3 = torch.cat((data[:, 6:9, :], data[:, 15:18, :]), dim=1)
            h, w = 112, 112
            m = torch.nn.ReplicationPad2d((h, h , w, w))
            data1 = m(data1)
            data2 = m(data2)
            data3 = m(data3)

            # padded1[:,:,12:1012, 18:1518:] = data1.transpose(2,3)
            # padded2[:, :, 12:1012, 18:1518:] = data2.transpose(2,3)
            # padded3[:, :, 12:1012, 18:1518:] = data3.transpose(2,3)
            data1 = Variable(data1, volatile=True)
            data2 = Variable(data2, volatile=True)
            data3 = Variable(data3, volatile=True)
            target = Variable(target, volatile=True)

            data1 = data1.transpose(2, 3)
            data2 = data2.transpose(2, 3)
            data3 = data3.transpose(2, 3)
            target = target.transpose(2, 3)
            # refield = 64
            # min_size = 512
            mode = 2
            refield = 64
            min_size = 64
            sf = 1
            modulo = 1
            # output = test_split(model, data1, data2, data3, refield, min_size, sf, modulo)
            output, = model(data1, data2, data3)

            # save the result to .H5 files
            hdrfile = h5py.File(args.result_dir + Test_Data_name + '_hdr.h5', 'w')
            img = output[0, :, :, :]
            img = tv.utils.make_grid(img.data.cpu()).numpy()
            hdrfile.create_dataset('data', data=img)
            hdrfile.close()

            # hdr = torch.log(1 + 5000 * output.cpu()) / torch.log(
            #     Variable(torch.from_numpy(np.array([1 + 5000])).float()))
            # target = torch.log(1 + 5000 * target).cpu() / torch.log(
            #     Variable(torch.from_numpy(np.array([1 + 5000])).float()))

            val = psnr(output, target)
            mu_val = psnr(range_compressor(output), range_compressor(target))
            test_loss += val
            mu_psnr += mu_val
        test_loss = test_loss / len(test_loaders.dataset)
        mu_psnr = mu_psnr / len(test_loaders.dataset)
        print('\n l-psnr:' + str(test_loss))
        print('\n mu-psnr:' + str(mu_psnr))
            # test_loss += F.mse_loss(hdr, target)
            # num = num + 1

        # test_loss = test_loss / len(test_loaders.dataset)
        # print('\n Test set: Average Loss: {:.4f}'.format(test_loss.data[0]))

    return test_loss


class testimage_dataloader(data.Dataset):
    def __init__(self, list_dir):
        f = open(list_dir)
        self.list_txt = f.readlines()
        self.length = len(self.list_txt)

    def __getitem__(self, index):
        # self.list_txt = self.list_txt[:-1]
        sample_path = self.list_txt[index][:-1]
        if os.path.exists(sample_path):
            f = h5py.File(sample_path, 'r')
            data = f['IN'][:]
            label = f['GT'][:]
            f.close()
        print(sample_path)
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return self.length

    def random_number(self, num):
        return random.randint(1, num)
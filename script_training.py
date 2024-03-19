import torch
import numpy as np
import time
import argparse
import torch.optim as optim
import torch.utils.data
import scipy.io as scio
from torch.nn import init
from dataset import DatasetFromHdf5

# from model_attention_trans import *
from running_func import *
from utils import *
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

parser = argparse.ArgumentParser(description='Attention-guided HDR')

parser.add_argument('--train-data', default='train_flow.txt')
parser.add_argument('--test_whole_Image', default='./test_flow.txt')
parser.add_argument('--trained_model_dir', default='./trained_model_vgg19/')
parser.add_argument('--trained_model_filename', default='PSNR_44.12_trained_model11749.pkl')
parser.add_argument('--result_dir', default='./result/')
parser.add_argument('--use_cuda', default=True)
parser.add_argument('--restore', default=False)
parser.add_argument('--load_model', default=False)

parser.add_argument('--lr', default=0.0001)
parser.add_argument('--seed', default=1)
parser.add_argument('--batchsize', default=4)
parser.add_argument('--epochs', default=800000)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--save_model_interval', default=1)

parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=6, help='number of color channels to use')

args = parser.parse_args()


torch.manual_seed(args.seed)
if args.use_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

#load data
train_loaders = torch.utils.data.DataLoader(
    data_loader(args.train_data),
    batch_size=args.batchsize, shuffle=True, num_workers=4)
testimage_dataset = torch.utils.data.DataLoader(
    testimage_dataloader(args.test_whole_Image),
    batch_size=1)

#make folders of trained model and result
mk_dir(args.result_dir)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data)


##
# model = model_PCD(args)
from model import ModelDeepHDR
# from model_3d_nonlocal import RDN_3D_6v
# model = ModelDeepHDR(in_nc=3,config=[4,4,4,4,4,4,4],dim=64)
model = ModelDeepHDR(in_nc=3,config=[4,4,4,4,4,4,4],dim=64)
# model = Generator()
hr_shape = (256, 256)

# netD = Discriminator()
# model = RDN_3D_6v()
# from model_ahdr import *
# model = AHDR(args)
model = nn.DataParallel(model)
# netD = nn.DataParallel(netD)

# model.apply(weights_init_kaiming)
# if args.use_cuda:
model.cuda()
# netD.cuda()

optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
# optimizer_D = optim.AdamW(netD.parameters(), lr=args.lr, betas=(0.9, 0.999))

# optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer,
                                                lr_lambda=lambda epoch: 0.95 ** epoch)
# schedulerD = optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer_D,
#                                                 lr_lambda=lambda epoch: 0.95 ** epoch)

##
start_step = 0
if args.restore and len(os.listdir(args.trained_model_dir)):
    model, start_step = model_restore(model, args.trained_model_dir)
    print('restart from {} step'.format(start_step))
# model = model_load(model, args.trained_model_dir, args.trained_model_filename)
# start_step = 11749
if __name__ == '__main__':
    # loss = testing_fun_1(model, testimage_dataset, args)
    for epoch in range(start_step + 1, args.epochs + 1):
        start = time.time()
        train(epoch, model, train_loaders, optimizer, args)
        end = time.time()
        
        scheduler.step()
        # schedulerD.step()
        print('epoch:{}, cost {} seconds'.format(epoch, end - start))

        if epoch % args.save_model_interval == 0:

            model_name = args.trained_model_dir + 'trained_model{}.pkl'.format(epoch)
            torch.save(model.state_dict(), model_name)
            # model_name = args.trained_model_dir + 'trained_model{}_D.pkl'.format(epoch)
            # torch.save(netD.state_dict(), model_name)

        if (epoch+1) % 1 == 0:
            loss = testing_fun_1(model, testimage_dataset, args)
            model_name = args.trained_model_dir + 'PSNR_' + str(round(loss, 2)) + '_trained_model{}.pkl'.format(epoch)
            torch.save(model.state_dict(), model_name)


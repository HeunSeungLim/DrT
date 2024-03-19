import time
import argparse
import torch.utils.data

from model import *
from running_func import *
from utils import *

parser = argparse.ArgumentParser(description='Attention-guided HDR')

parser.add_argument('--test_whole_Image', default='./test_flow.txt')
parser.add_argument('--trained_model_dir', default='./trained_model/')
parser.add_argument('--trained_model_filename', default='PSNR_44.42_trained_model11639.pkl')
parser.add_argument('--result_dir', default='./230305_res/')
parser.add_argument('--use_cuda', default=True)
parser.add_argument('--load_model', default=True)
parser.add_argument('--lr', default=0.000001)
parser.add_argument('--seed', default=1)

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#
# model = ModelDeepHDR(in_nc=3,config=[4,4,4,4,4,4,4],dim=64)
# if torch.cuda.device_count() > 1:
#   model = nn.DataParallel(model)
# from model_attention_3d_lee2 import mk4_HL
# from inpaint_g import BaseConvGenerator
# model = mk4_HL()


from model import ModelDeepHDR
# from model_3d_nonlocal import RDN_3D_6v
# model = ModelDeepHDR(in_nc=3,config=[4,4,4,4,4,4,4],dim=64)
model = ModelDeepHDR(in_nc=3,config=[4,4,4,4,4,4,4],dim=64)
# model = Generator()
hr_shape = (256, 256)
model = nn.DataParallel(model)
if args.use_cuda:
    model.cuda()

model.to(device)#

testimage_dataset = torch.utils.data.DataLoader(
    testimage_dataloader(args.test_whole_Image),
    batch_size=1)


#make folders of trained model and result
mk_dir(args.result_dir)
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
##
# model = AHDR(args)
# model.apply(weights_init_kaiming)


##
start_step = 0
# if args.load_model and len(os.listdir(args.trained_model_dir)):
model = model_load(model, args.trained_model_dir, args.trained_model_filename)

# In the testing, we test on the whole image, so we defind a new variable
#  'Image_test_loaders' used to load the whole image
start = time.time()
loss = testing_fun_1(model, testimage_dataset, args)
end = time.time()
print('Running Time: {} seconds'.format(end - start))

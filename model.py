import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath
# from bam import *
# from cbam import *

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        # self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(ffted)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output

class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(
            self.relative_position_params.view(2 * window_size - 1, 2 * window_size - 1, self.n_heads).transpose(1,
                                                                                                                 2).transpose(
                0, 1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type != 'W': x = torch.roll(x, shifts=(-(self.window_size // 2), -(self.window_size // 2)), dims=(1, 2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        # assert h_windows == w_windows

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size // 2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type != 'W': output = torch.roll(output, shifts=(self.window_size // 2, self.window_size // 2),
                                                 dims=(1, 2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        # negative is allowed
        return self.relative_position_params[:, relation[:, :, 0].long(), relation[:, :, 1].long()]


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        # print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x



class SP_layer(nn.Module):
    def init(self,in_feature,out_feature,radius):
        super(SP_layer,self).init()
        self.radius=radius
        # self.linear=nn.Linear(in_feature-1,out_feature-1)
        # self.dropout=nn.Dropout(0.5)
        # self.apply(init_weights)

    def forward(self, x):
        v = self.log(x, r=self.radius)
        # v = self.linear(v)
        # v = self.dropout(v)
        x = self.exp(v, r=self.radius)
        x = self.srelu(x, r=self.radius)
        return x

    def exp(self,v, o=None, r=1.0):
        if v.is_cuda == True:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        if o is None:
            _, c, h, w = x.size()
            # o = torch.cat([torch.zeros(1, c, h - 1, w), r * torch.ones(1, c, 1, 1)], dim=2).to(device)
            o = torch.cat([torch.zeros(1, v.size(1)), r * torch.Tensor([[1]])], dim=1).to(device)
        theta = torch.norm(v, dim=1, keepdim=True) / r
        v = torch.cat([v, torch.zeros(v.size(0), 1,256,256).to(device)], dim=1)
        return torch.cos(theta) * o.unsqueeze(2).unsqueeze(2) + torch.sin(theta) * F.normalize(v, dim=1) * r

    def log(self,x, o=None, r=1.0):
        if x.is_cuda == True:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        if o is None:
            o = torch.cat([torch.zeros(1, x.size(1) - 1), r * torch.Tensor([[1]])], dim=1).to(device)
        # c = F.cosine_similarity(x, o, dim=1).view(-1, 1)
        c = F.cosine_similarity(x, o.unsqueeze(2).unsqueeze(2), dim=1)
        theta = torch.acos(self.shrink(c))
        # v = F.normalize(x - c * o, dim=1)[:, :-1]
        v = F.normalize(x - o.unsqueeze(2).unsqueeze(2) * o.unsqueeze(2).unsqueeze(2), dim = 1)[:, :-1]
        return r * theta * v

    def shrink(self,x, epsilon=1e-4):
        x[torch.abs(x) > (1 - epsilon)] = x[torch.abs(x) > (1 - epsilon)] * (1 - epsilon)
        return x

    def srelu(self,x, r=1.0):
        return r * F.normalize(F.relu(x), dim=1)

class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution

        assert self.type in ['W', 'SW']
        if self.input_resolution <= self.window_size:
            self.type = 'W'

        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path,
                                 self.type, self.input_resolution)
        self.conv1_1 = nn.Conv2d(self.conv_dim + self.trans_dim+1, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim+1, 1, 1, 0, bias=True)

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False)
        )

        self.conv_trans1 = nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False)
        self.conv_trans2 = nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False)

        self.conv_Rconv1 = nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False)
        self.spectral_transfrom = SpectralTransform(in_channels = self.conv_dim, out_channels = self.conv_dim)

        self.relu = nn.ReLU(True)


        ############ spherical mapping

        


    def exp(self,v, o=None, r=1.0):
        if v.is_cuda == True:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        if o is None:
            _, c, h, w = x.size()
            # o = torch.cat([torch.zeros(1, c, h - 1, w), r * torch.ones(1, c, 1, 1)], dim=2).to(device)
            o = torch.cat([torch.zeros(1, v.size(1)), r * torch.Tensor([[1]])], dim=1).to(device)
        theta = torch.norm(v, dim=1, keepdim=True) / r
        v = torch.cat([v, torch.zeros(v.size(0), 1, v.size(2), v.size(3)).to(device)], dim=1)
        return torch.cos(theta) * o.unsqueeze(2).unsqueeze(2) + torch.sin(theta) * F.normalize(v, dim=1) * r

    def log(self,x, o=None, r=1.0):
        if x.is_cuda == True:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        if o is None:
            o = torch.cat([torch.zeros(1, x.size(1) - 1), r * torch.Tensor([[1]])], dim=1).to(device)
        # c = F.cosine_similarity(x, o, dim=1).view(-1, 1)
        c = F.cosine_similarity(x, o.unsqueeze(2).unsqueeze(2), dim=1)
        theta = torch.acos(self.shrink(c))
        # v = F.normalize(x - c * o, dim=1)[:, :-1]
        v = F.normalize(x - o.unsqueeze(2).unsqueeze(2) * o.unsqueeze(2).unsqueeze(2), dim = 1)[:, :-1]
        return r * theta * v

    def shrink(self,x, epsilon=1e-4):
        x[torch.abs(x) > (1 - epsilon)] = x[torch.abs(x) > (1 - epsilon)] * (1 - epsilon)
        return x

    def srelu(self,x, r=1.0):
        return r * F.normalize(F.relu(x), dim=1)


    def forward(self, x):
        residual = x

        x = self.exp(x, r=10)
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)

        temp_trans1 = self.conv_trans1(trans_x)
        temp_trans2 = self.conv_trans2(trans_x)

        temp_conv1 = self.conv_trans1(conv_x)
        temp_conv2 = self.spectral_transfrom(conv_x)

        temp_out1 = self.relu(temp_trans1 + temp_conv1)
        temp_out2 = self.relu(temp_trans2 + temp_conv2)

        # res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        res = self.conv1_2(torch.cat((temp_out1, temp_out2), dim=1))
        # x =  res
        x = self.log(res) + residual
        return x


class make_dilation_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dilation_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2+1, bias=True, dilation=2)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

# Dilation Residual dense block (DRDB)
class DRDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(DRDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):    
        modules.append(make_dilation_dense(nChannels_, growthRate))
        nChannels_ += growthRate 
    self.dense_layers = nn.Sequential(*modules)    
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out

# Attention Guided HDR, AHDR-Net
class AHDR(nn.Module):
    def __init__(self, args):
        super(AHDR, self).__init__()
        nChannel = args.nChannel
        nDenselayer = args.nDenselayer
        nFeat = args.nFeat
        growthRate = args.growthRate
        self.args = args

        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat*3, nFeat, kernel_size=3, padding=1, bias=True)
        self.att11 = nn.Conv2d(nFeat*2, nFeat*2, kernel_size=3, padding=1, bias=True)
        self.att12 = nn.Conv2d(nFeat*2, nFeat, kernel_size=3, padding=1, bias=True)
        self.attConv1 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.att31 = nn.Conv2d(nFeat*2, nFeat*2, kernel_size=3, padding=1, bias=True)
        self.att32 = nn.Conv2d(nFeat*2, nFeat, kernel_size=3, padding=1, bias=True)
        self.attConv3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # DRDBs 3
        self.RDB1 = DRDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = DRDB(nFeat, nDenselayer, growthRate)

        self.RDB3 = DRDB(nFeat, nDenselayer, growthRate)
        # feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # fusion
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # conv 
        self.conv3 = nn.Conv2d(nFeat, 3, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()


    def forward(self, x1, x2, x3):

        F1_ = self.relu(self.conv1(x1))
        F2_ = self.relu(self.conv1(x2))
        F3_ = self.relu(self.conv1(x3))

        F1_i = torch.cat((F1_, F2_), 1)
        F1_A = self.relu(self.att11(F1_i))
        F1_A = self.att12(F1_A)
        F1_A = nn.functional.sigmoid(F1_A)
        F1_ = F1_ * F1_A


        F3_i = torch.cat((F3_, F2_), 1)
        F3_A = self.relu(self.att31(F3_i))
        F3_A = self.att32(F3_A)
        F3_A = nn.functional.sigmoid(F3_A)
        F3_ = F3_ * F3_A

        F_ = torch.cat((F1_, F2_, F3_), 1)

        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        FF = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(FF)         
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F2_
        us = self.conv_up(FDF)

        output = self.conv3(us)
        output = nn.functional.sigmoid(output)

        return output




class ModelDeepHDR(nn.Module):
    def __init__(self,  in_nc=3, config=[2, 2, 2, 2, 2, 2, 2], dim=64, drop_path_rate=0.0, input_resolution=64):
        super(ModelDeepHDR, self).__init__()

        # F-1
        in_channel = 6
        nFeat = 64
        # self.conv1 = nn.Conv2d(in_channel, nFeat, kernel_size=3, padding=1, bias=True)
        # # F0
        # self.conv2 = nn.Conv2d(nFeat * 3, nFeat, kernel_size=3, padding=1, bias=True)
        # self.att11 = nn.Conv2d(nFeat * 2, nFeat * 2, kernel_size=3, padding=1, bias=True)
        # self.att12 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        # self.attConv1 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # self.att31 = nn.Conv2d(nFeat * 2, nFeat * 2, kernel_size=3, padding=1, bias=True)
        # self.att32 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        # self.attConv3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        self.config = config
        self.dim = dim
        self.head_dim = 32
        self.window_size = 8
        nFeat = dim
        # dim = 32

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = [nn.Conv2d(nFeat*3, dim, 3, 1, 1, bias=False)]

        begin = 0
        self.m_down1 = [ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution)
                        for i in range(config[0])] + \
                       [nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution // 2)
                        for i in range(config[1])] + \
                       [nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down3 = [ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution // 4)
                        for i in range(config[2])] + \
                       [nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_body = [ConvTransBlock(4 * dim, 4 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                      'W' if not i % 2 else 'SW', input_resolution // 8)
                       for i in range(config[3])]

        begin += config[3]
        self.m_up3 = [nn.ConvTranspose2d(8 * dim, 4 * dim, 2, 2, 0, bias=False), ] + \
                     [ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW', input_resolution // 4)
                      for i in range(config[4])]

        begin += config[4]
        self.m_up2 = [nn.ConvTranspose2d(4 * dim, 2 * dim, 2, 2, 0, bias=False), ] + \
                     [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW', input_resolution // 2)
                      for i in range(config[5])]

        begin += config[5]
        self.m_up1 = [nn.ConvTranspose2d(2 * dim, dim, 2, 2, 0, bias=False), ] + \
                     [ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW', input_resolution)
                      for i in range(config[6])]

        self.m_tail = [nn.Conv2d(dim, in_nc, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.m_tail = nn.Sequential(*self.m_tail)

        self.relu = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(6, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat * 3, nFeat, kernel_size=3, padding=1, bias=True)
        self.att11 = nn.Conv2d(nFeat * 2, nFeat * 2, kernel_size=3, padding=1, bias=True)
        self.att12 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        self.attConv1 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.att31 = nn.Conv2d(nFeat * 2, nFeat * 2, kernel_size=3, padding=1, bias=True)
        self.att32 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        self.attConv3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # self.bam1 = BAM(nFeat*2)
        # self.bam2 = BAM(nFeat*2)
        # self.bam3 = BAM(nFeat*2)

    def forward(self,  x1, x2, x3):
        h, w = x1.size()[-2:]
        paddingBottom = int(np.ceil(h / 64) * 64 - h)
        paddingRight = int(np.ceil(w / 64) * 64 - w)
        x1 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x1)
        x2 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x2)
        x3 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x3)

        F1_ = self.relu(self.conv1(x1))
        F2_ = self.relu(self.conv1(x2))
        F3_ = self.relu(self.conv1(x3))
        F1_i = torch.cat((F1_, F2_), 1)
        F1_A = self.relu(self.att11(F1_i))
        F1_A = self.att12(F1_A)
        F1_A = nn.functional.sigmoid(F1_A)
        F1_ = F1_ * F1_A
        F3_i = torch.cat((F3_, F2_), 1)
        F3_A = self.relu(self.att31(F3_i))
        F3_A = self.att32(F3_A)
        F3_A = nn.functional.sigmoid(F3_A)
        F3_ = F3_ * F3_A
        F_ = torch.cat((F1_, F2_, F3_), 1)
        # F_ = torch.cat((F1_, F2_, F3_), 1)

        x1 = self.m_head(F_)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2) + F2_
        x = self.m_tail(x + x1)


        # imgs = self.post_convolution_steps(out, **inputs)
        x = nn.functional.sigmoid(x)

        x = x[..., :h, :w]
        # F3_A = F3_A[..., :h, :w]
        # F1_A = F1_A[..., :h, :w]

        return x
    


class ModelDeepHDR_nonglobal(nn.Module):
    def __init__(self,  in_nc=3, config=[2, 2, 2, 2, 2, 2, 2], dim=64, drop_path_rate=0.0, input_resolution=64):
        super(ModelDeepHDR_nonglobal, self).__init__()

        # F-1
        in_channel = 6
        nFeat = 64
        # self.conv1 = nn.Conv2d(in_channel, nFeat, kernel_size=3, padding=1, bias=True)
        # # F0
        # self.conv2 = nn.Conv2d(nFeat * 3, nFeat, kernel_size=3, padding=1, bias=True)
        # self.att11 = nn.Conv2d(nFeat * 2, nFeat * 2, kernel_size=3, padding=1, bias=True)
        # self.att12 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        # self.attConv1 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # self.att31 = nn.Conv2d(nFeat * 2, nFeat * 2, kernel_size=3, padding=1, bias=True)
        # self.att32 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        # self.attConv3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        self.config = config
        self.dim = dim
        self.head_dim = 32
        self.window_size = 8
        nFeat = dim
        # dim = 32

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = [nn.Conv2d(nFeat*3, dim, 3, 1, 1, bias=False)]

        begin = 0
        self.m_down1 = [ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution)
                        for i in range(config[0])] + \
                       [nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution // 2)
                        for i in range(config[1])] + \
                       [nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down3 = [ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution // 4)
                        for i in range(config[2])] + \
                       [nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_body = [ConvTransBlock(4 * dim, 4 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                      'W' if not i % 2 else 'SW', input_resolution // 8)
                       for i in range(config[3])]

        begin += config[3]
        self.m_up3 = [nn.ConvTranspose2d(8 * dim, 4 * dim, 2, 2, 0, bias=False), ] + \
                     [ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW', input_resolution // 4)
                      for i in range(config[4])]

        begin += config[4]
        self.m_up2 = [nn.ConvTranspose2d(4 * dim, 2 * dim, 2, 2, 0, bias=False), ] + \
                     [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW', input_resolution // 2)
                      for i in range(config[5])]

        begin += config[5]
        self.m_up1 = [nn.ConvTranspose2d(2 * dim, dim, 2, 2, 0, bias=False), ] + \
                     [ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW', input_resolution)
                      for i in range(config[6])]

        self.m_tail = [nn.Conv2d(dim, in_nc, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.m_tail = nn.Sequential(*self.m_tail)

        self.relu = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(6, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat * 3, nFeat, kernel_size=3, padding=1, bias=True)
        self.att11 = nn.Conv2d(nFeat * 2, nFeat * 2, kernel_size=3, padding=1, bias=True)
        self.att12 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        self.attConv1 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.att31 = nn.Conv2d(nFeat * 2, nFeat * 2, kernel_size=3, padding=1, bias=True)
        self.att32 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        self.attConv3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # self.bam1 = BAM(nFeat*2)
        # self.bam2 = BAM(nFeat*2)
        # self.bam3 = BAM(nFeat*2)

    def forward(self,  x1, x2, x3):
        h, w = x1.size()[-2:]
        paddingBottom = int(np.ceil(h / 64) * 64 - h)
        paddingRight = int(np.ceil(w / 64) * 64 - w)
        x1 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x1)
        x2 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x2)
        x3 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x3)

        F1_ = self.relu(self.conv1(x1))
        F2_ = self.relu(self.conv1(x2))
        F3_ = self.relu(self.conv1(x3))
        F1_i = torch.cat((F1_, F2_), 1)
        F1_A = self.relu(self.att11(F1_i))
        F1_A = self.att12(F1_A)
        F1_A = nn.functional.sigmoid(F1_A)
        F1_ = F1_ * F1_A
        F3_i = torch.cat((F3_, F2_), 1)
        F3_A = self.relu(self.att31(F3_i))
        F3_A = self.att32(F3_A)
        F3_A = nn.functional.sigmoid(F3_A)
        F3_ = F3_ * F3_A
        F_ = torch.cat((F1_, F2_, F3_), 1)
        # F_ = torch.cat((F1_, F2_, F3_), 1)

        x1 = self.m_head(F_)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)


        # imgs = self.post_convolution_steps(out, **inputs)
        x = nn.functional.sigmoid(x)
        x = x[..., :h, :w]
        F3_A = F3_A[..., :h, :w]
        F1_A = F1_A[..., :h, :w]

        return x, F1_A, F3_A



if __name__ == '__main__':

    x = torch.randn(1, 6, 256, 256)

    model = ModelDeepHDR(in_nc=3,config=[4,4,4,4,4,4,4],dim=64)
    print(model)
    output = model(x,x,x)
    print(f'output: {output.shape}')

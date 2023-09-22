import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
from utils import *
import matplotlib.pyplot as plt

class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)

        # (N, C, HW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)  # (N,HW,C)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)  # (N,HW,C)
            f = torch.matmul(theta_x, phi_x)  # (N,HW,C)(N,C,HW)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # (N,C,HW)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # (N,C,HW)
            theta_x = theta_x.permute(0, 2, 1)  # (N,HW,C)
            f = torch.matmul(theta_x, phi_x)  # (N,HW,C)(N,C,HW)-->(N,HW)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)  # (N,HW)(N,HW,C)-->(N,HW,C)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()  # (N,C,HW)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # (N,C,H,W)

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z


class ResnetGenerator(nn.Module):
    # Generator architecture
    def __init__(self, input_nc=3, output_nc=3, inter_nc=64, n_blocks=6, img_size=32, use_bias=False, rs_norm='BN',
                 padding_type='zero', dsple=False, scale_factor=4, non_local=False):

        super(ResnetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.inter_nc = inter_nc
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.use_bias = use_bias
        self.rs_norm = rs_norm
        self.padding_type = padding_type
        self.dsple = dsple
        self.scale_factor = scale_factor

        # Input blocks
        InBlock = []

        InBlock += [nn.Conv2d(input_nc, inter_nc, kernel_size=7, stride=1, padding=3, bias=self.use_bias),
                    nn.LeakyReLU(0.2)]
        InBlock += [
            nn.Conv2d(inter_nc, inter_nc, kernel_size=3, stride=2 if self.dsple and self.scale_factor == 4 else 1,
                      padding=1, bias=self.use_bias),
            nn.LeakyReLU(0.2)]  # changed
        InBlock += [
            nn.Conv2d(inter_nc, inter_nc, kernel_size=3, stride=2 if self.dsple else 1, padding=1, bias=self.use_bias),
            nn.LeakyReLU(0.2)]

        # ResnetBlocks
        ResnetBlocks = []

        for i in range(n_blocks):
            ResnetBlocks += [ResnetBlock(inter_nc, self.padding_type, self.rs_norm, self.use_bias)]

        # add non-local block after layer 2
        self.layer2 = self._make_layer(ResnetBlocks, 32, stride=2, non_local=non_local)
        self.layer3 = self._make_layer(ResnetBlocks, 64, stride=2)

        # Output block
        OutBlock = []

        OutBlock += [nn.Conv2d(inter_nc, inter_nc, kernel_size=3, stride=1, padding=1, bias=self.use_bias),
                     nn.LeakyReLU(0.2)]
        OutBlock += [nn.Conv2d(inter_nc, inter_nc, kernel_size=3, stride=1, padding=1, bias=self.use_bias),
                     nn.LeakyReLU(0.2)]
        OutBlock += [nn.Conv2d(inter_nc, output_nc, kernel_size=7, stride=1, padding=3, bias=self.use_bias),
                     nn.LeakyReLU(0.2)]

        self.InBlock = nn.Sequential(*InBlock)
        self.ResnetBlocks = nn.Sequential(*ResnetBlocks)
        self.OutBlock = nn.Sequential(*OutBlock)

    def _make_layer(self, block, planes, stride, non_local=True):
        strides = [stride]
        layers = []

        last_idx = len(strides)

        if non_local:
            layers.append(NLBlockND(in_channels=64, dimension=2))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.InBlock(x)
        out = self.ResnetBlocks(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.OutBlock(out)

        return out


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_type, use_bias):
        # dim(int) -- The number of channels in the resnet blocks
        # padding_type(str) -- The name of padding layer: reflect | replicate | zero
        # norm_type(str) -- The type of normalization method. BN : Batch Normalization, IN : Instance Normalization, else : none
        # use_bias -- Whether to use bias on conv layer or not
        super(ResnetBlock, self).__init__()

        conv_block = []

        # Padding
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if norm_type == 'BN':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'IN':
            norm_layer = nn.InstanceNorm2d
        else:
            raise NotImplementedError('Normalization [%s] is not implemented' % norm_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.LeakyReLU(0.2)]

        # Padding
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.LeakyReLU(0.2)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)

        # Skip connection
        out = out + x

        return out


class Discriminator(nn.Module):
    def __init__(self, input_nc=3, norm_type='BN', use_bias=True, is_inner=True, scale_factor=4):
        # input_nc(int) -- The number of channels of input img
        # norm_type(str) -- The type of normalization method. BN : Batch Normalization, IN : Instance Normalization, else : none
        # use_bias(bool) -- Whether to use bias or not
        # is_inner(bool) -- True : For inner cycle, False : For outer cycle
        # scale_factor(int) -- Scale factor, 2 / 4

        super(Discriminator, self).__init__()

        if norm_type == 'BN':
            norm_layer = nn.BatchNorm2d
            use_bias = False  # There is no need to use bias because BN already has shift parameter.
        elif norm_type == 'IN':
            norm_layer = nn.InstanceNorm2d
        else:
            raise NotImplementedError('Normalization [%s] is not implemented' % norm_type)

        if is_inner == True:
            s = 1
        elif is_inner == False:
            s = 2
        else:
            raise NotImplementedError('is_inner must be boolean.')

        nfil_mul = 64
        p = 0  # Why 1???
        layers = []
        layers += [
            nn.Conv2d(input_nc, nfil_mul, kernel_size=4, stride=2 if is_inner == True and scale_factor == 2 else s,
                      padding=p, bias=use_bias),
            nn.LeakyReLU(0.2)]  # changed
        layers += [nn.Conv2d(nfil_mul, nfil_mul * 2, kernel_size=4, stride=s, padding=p, bias=use_bias),
                   norm_layer(nfil_mul * 2),
                   nn.LeakyReLU(0.2)]
        layers += [nn.Conv2d(nfil_mul * 2, nfil_mul * 4, kernel_size=4, stride=s, padding=p, bias=use_bias),
                   norm_layer(nfil_mul * 4),
                   nn.LeakyReLU(0.2)]
        layers += [nn.Conv2d(nfil_mul * 4, nfil_mul * 8, kernel_size=4, stride=1, padding=p, bias=use_bias),
                   norm_layer(nfil_mul * 8),
                   nn.LeakyReLU(0.2)]
        layers += [nn.Conv2d(nfil_mul * 8, 1, kernel_size=4, stride=1, padding=p, bias=use_bias),
                   nn.LeakyReLU(0.2)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)

        return out  # Predicted values of each patches


def test():


    # device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load image
    img = pil_loader("data/coal2D_train_HR/0001.png")

    # Transform image to torch Tensor. Normalized to 0~1 automatically.
    img = transforms.Resize((128, 128))(img)
    img = transforms.ToTensor()(img).to(device)

    print("Input shape : ", img.shape)

    # Feed to generator
    img = torch.unsqueeze(img, 0)
    G1 = ResnetGenerator(dsple=True).to(device)
    fakeimgs = G1(img)

    print("Fake image shape : ", fakeimgs.shape)

    # Feed to discriminator
    D1 = Discriminator(is_inner=True).to(device)
    out = D1(fakeimgs)
    print(out.shape)
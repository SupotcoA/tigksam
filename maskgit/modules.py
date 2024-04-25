import torch
from torch import nn
import torch.nn.functional as F


def cf2cl(tensor):
    return torch.permute(tensor, [0, 2, 3, 1])


def cl2cf(tensor):
    return torch.permute(tensor, [0, 3, 1, 2])


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3):
        super().__init__()
        out_channels = out_channels if out_channels else in_channels

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=2,
                              padding=1)

    def forward(self, x):
        h = self.conv(x)
        return h


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3):
        super().__init__()
        out_channels = out_channels if out_channels else in_channels

        if kernel_size == 4:
            self.upsample = nn.Identity()
            self.conv = nn.ConvTranspose2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1,
                                           output_padding=0)

        elif kernel_size == 3:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            self.conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding='same')

    def forward(self, x):
        h = self.upsample(x)
        h = self.conv(h)
        return h


class ResBlock(nn.Module):

    def __init__(self, in_channels,
                 bottle_neck_channels=None,
                 out_channels=None,
                 res_bottle_neck_factor=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels
        if bottle_neck_channels is not None:
            self.bottle_neck_channels = bottle_neck_channels
        else:
            self.bottle_neck_channels = max(self.out_channels,
                                            self.in_channels) \
                                        // res_bottle_neck_factor
            self.bottle_neck_channels = max(32, self.bottle_neck_channels)
        if self.bottle_neck_channels >= 128 and divmod(self.bottle_neck_channels,32)[1]!=0:
            self.bottle_neck_channels = (1+int(self.bottle_neck_channels/32))*32

        self.norm1 = nn.GroupNorm(num_groups=min(32, in_channels//4),
                                  num_channels=in_channels,
                                  eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=self.bottle_neck_channels,
                               kernel_size=3,
                               stride=1,
                               padding='same',
                               bias=False)
        self.norm2 = nn.GroupNorm(num_groups=min(32, self.bottle_neck_channels//4),
                                  num_channels=self.bottle_neck_channels,
                                  eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(in_channels=self.bottle_neck_channels,
                               out_channels=self.out_channels,
                               kernel_size=3,
                               stride=1,
                               padding='same',
                               bias=False)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels=in_channels,
                                           out_channels=self.out_channels,
                                           kernel_size=1,
                                           stride=1,
                                           padding='same')
        else:
            self.conv_shortcut = nn.Identity()
        self.rescale = 1  # / config['num_res_blocks']

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.relu_(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.relu_(h)
        h = self.conv2(h)
        x = self.conv_shortcut(x)

        return x + h  # self.rescale


class AttnBlock(nn.Module):
    def __init__(self, in_channels, embed_channels=None):
        super().__init__()
        self.in_channels = in_channels
        if embed_channels is not None:
            self.embed_channels = embed_channels
        else:
            self.embed_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels,
                                 eps=1e-6, affine=False)
        self.q = nn.Conv2d(in_channels,
                           self.embed_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.k = nn.Conv2d(in_channels,
                           self.embed_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.v = nn.Conv2d(in_channels,
                           self.embed_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.proj_out = nn.Conv2d(self.embed_channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):

    def __init__(self, in_channels=3,
                 conv_in_channels=64,
                 out_channels=256,
                 channels_mult=(1, 1, 2, 2, 4),
                 num_res_block=1):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels=in_channels,
                                 out_channels=conv_in_channels * channels_mult[0],
                                 kernel_size=3,
                                 stride=1,
                                 padding='same')
        current_channels = conv_in_channels * channels_mult[0]

        layers = nn.ModuleList()
        for i, m in enumerate(channels_mult):
            blk_in = current_channels
            blk_out = conv_in_channels * m
            if i != len(channels_mult) - 1:
                for _ in range(num_res_block):
                    layers.append(ResBlock(in_channels=blk_in,
                                           out_channels=blk_in))
                layers.append(DownSample(in_channels=blk_in,
                                         out_channels=blk_out))

            else:
                layers.append(ResBlock(in_channels=blk_in,
                                       out_channels=blk_out))
            current_channels = blk_out
        self.layers = layers

        self.mid_res1 = ResBlock(in_channels=current_channels,
                                 out_channels=current_channels)

        self.mid_attn = nn.Identity()  # AttnBlock(in_channels=current_channels)

        self.mid_res2 = ResBlock(in_channels=current_channels,
                                 out_channels=current_channels)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=current_channels,
                                     eps=1e-6, affine=False)
        self.pre_vq_conv = nn.Conv2d(in_channels=current_channels,
                                     out_channels=out_channels,
                                     kernel_size=1,
                                     stride=1,
                                     padding='same')

    def forward(self, x):
        h = self.conv_in(x)
        for layer in self.layers:
            h = layer(h)
        h = self.mid_res1(h)
        h = self.mid_attn(h)
        h = self.mid_res2(h)
        h = self.norm_out(h)
        h = F.relu_(h)
        h = self.pre_vq_conv(h)
        return h


class Decoder(nn.Module):

    def __init__(self, in_channels=256,
                 conv_in_channels=64,
                 channels_mult=(1, 1, 2, 2, 4),
                 out_channels=3,
                 num_res_block=1):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels=in_channels,
                                 out_channels=conv_in_channels * channels_mult[-1],
                                 kernel_size=3,
                                 stride=1,
                                 padding='same')
        current_channels = conv_in_channels * channels_mult[-1]

        self.mid_res1 = ResBlock(in_channels=current_channels,
                                 out_channels=current_channels)

        self.mid_attn = nn.Identity()  # AttnBlock(in_channels=current_channels)

        self.mid_res2 = ResBlock(in_channels=current_channels,
                                 out_channels=current_channels)

        layers = nn.ModuleList()
        for i, m in enumerate(reversed((1,) + channels_mult[:-1])):
            blk_in = current_channels
            blk_out = conv_in_channels * m
            if i != 0:
                for _ in range(num_res_block-1):
                    layers.append(ResBlock(in_channels=blk_in,
                                           out_channels=blk_in))
                layers.append(ResBlock(in_channels=blk_in,
                                       out_channels=blk_out))  ###
                layers.append(UpSample(in_channels=blk_out,
                                       out_channels=blk_out))

            else:
                layers.append(ResBlock(in_channels=blk_in,
                                       out_channels=blk_out))
            current_channels = blk_out
        self.layers = layers

        self.norm_out = nn.GroupNorm(num_groups=min(32, current_channels//4),
                                     num_channels=current_channels,
                                     eps=1e-6, affine=False)
        self.conv_out = nn.Conv2d(in_channels=current_channels,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding='same')

    def forward(self, x):
        h = self.conv_in(x)
        h = self.mid_res1(h)
        h = self.mid_attn(h)
        h = self.mid_res2(h)
        for layer in self.layers:
            h = layer(h)
        h = self.norm_out(h)
        h = F.relu_(h)
        h = self.conv_out(h)
        return h


class Discriminator(nn.Module):

    def __init__(self, in_channels=3,
                 conv_in_channels=64,
                 out_channels=1,
                 channels_mult=(1, 2, 4)):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels=in_channels,
                                 out_channels=conv_in_channels * channels_mult[0],
                                 kernel_size=4,
                                 stride=2,
                                 padding=1)
        current_channels = conv_in_channels * channels_mult[0]
        self.conv_in_activation = nn.LeakyReLU(0.2, True)

        layers = nn.ModuleList()
        for i, m in enumerate(channels_mult[1:]):
            blk_in = current_channels
            blk_out = conv_in_channels * m
            layers.append(nn.Conv2d(in_channels=blk_in,
                                    out_channels=blk_out,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1,
                                    bias=False))
            layers.append(nn.GroupNorm(num_groups=min(32, blk_out//4),
                                       num_channels=blk_out,
                                       eps=1e-6, affine=True))  # need norm
            layers.append(nn.LeakyReLU(0.2, True))
            current_channels = blk_out

        layers.append(nn.Conv2d(in_channels=current_channels,
                                out_channels=current_channels,
                                kernel_size=4,
                                stride=1,
                                padding=1,
                                bias=True))
        layers.append(nn.GroupNorm(num_groups=min(32, current_channels//4),
                                   num_channels=current_channels,
                                   eps=1e-6, affine=True))
        layers.append(nn.LeakyReLU(0.2, True))

        self.layers = layers

        self.conv_out = nn.Conv2d(in_channels=current_channels,
                                  out_channels=out_channels,
                                  kernel_size=4,
                                  stride=1,
                                  padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        h = self.conv_in_activation(h)
        for layer in self.layers:
            h = layer(h)
        h = self.conv_out(h)
        return h

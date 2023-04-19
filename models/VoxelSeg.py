import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=True),
            nn.BatchNorm3d(num_features=out_channels),
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=True),
            nn.BatchNorm3d(num_features=out_channels),
            nn.LeakyReLU(inplace=False),
        )

    def forward(self, x):
        return self.conv(x)


class ConvEncoder3D(nn.Module):
    def __init__(self, in_channels, out_channels, pool_kernel=(2, 2, 2), pool_stride=(2, 2, 2)):
        super(ConvEncoder3D, self).__init__()

        self.encoder = nn.Sequential(
            ConvBlock3D(in_channels=in_channels, out_channels=out_channels),
            nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride)
        )

    def forward(self, x):
        return self.encoder(x)


class ConvDecoder3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvDecoder3D, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=True)
        self.conv = ConvBlock3D(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        up_depth, up_height, up_width = x.size()[2:]
        sk_depth, sk_height, sk_width = skip.size()[2:]

        diff_z = sk_depth - up_depth
        diff_y = sk_height - up_height
        diff_x = sk_width - up_width

        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2,
                      diff_z // 2, diff_z - diff_z // 2])

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class Unet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channels=[32, 64, 128]):
        super(Unet3D, self).__init__()

        self.channels = channels
        self.num_layers = len(self.channels) - 1

        self.incoder = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=channels[0] // 2,
                      kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=True),
            nn.LeakyReLU(inplace=False)
        )

        self.encoders = nn.ModuleList()
        for index, channel in enumerate(channels):
            self.encoders.append(
                ConvEncoder3D(in_channels=channels[index - 1] if index > 0 else channels[0] // 2,
                              out_channels=channel)
            )

        self.decoders = nn.ModuleList()
        for index, channel in enumerate(channels):
            index = self.num_layers - index
            self.decoders.append(
                ConvDecoder3D(in_channels=channels[index],
                              out_channels=channels[index - 1] if index > 0 else channels[0] // 2)
            )

        self.outcoder = nn.Conv3d(in_channels=channels[0] // 2, out_channels=out_channels,
                                  kernel_size=(1, 1, 1), stride=(1, 1, 1))

    def forward(self, image):
        x = self.incoder(image)
        skipes = [x]
        for index, encoder in enumerate(self.encoders):
            skipes.append(encoder(skipes[index]))

        outputs = []
        for index, decoder in enumerate(self.decoders):
            x = decoder(x=x if index > 0 else skipes[-1], skip=skipes[-index - 2])
            outputs.append(torch.sigmoid(x))

        x = torch.sigmoid(self.outcoder(x))
        return x, outputs


def create_model(in_channels=1, out_channels=1, channels=[16, 32, 64], *args):
    unet = Unet3D(in_channels=in_channels, out_channels=out_channels, channels=channels)
    return unet

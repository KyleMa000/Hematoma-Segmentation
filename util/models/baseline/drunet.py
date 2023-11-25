import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, need_activate=True):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding='same')
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ELU() if need_activate else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.activation(out)
        return out

class Block1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block1, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels // 4, kernel_size=1)
        self.conv2 = ConvLayer(out_channels // 4, out_channels // 4, kernel_size=3)
        self.conv3 = ConvLayer(out_channels // 4, out_channels // 4, kernel_size=3, need_activate=False)
        self.residual = ConvLayer(in_channels, out_channels // 4, kernel_size=1)
        self.elu = nn.ELU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        res = self.residual(x)
        out = self.elu(out + res)
        return out


class Block3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block3, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels // 4, kernel_size=1)
        self.conv2 = ConvLayer(out_channels // 4, out_channels // 4, kernel_size=3)
        self.conv3 = ConvLayer(out_channels // 4, out_channels, kernel_size=3, need_activate=False)
        self.residual = ConvLayer(in_channels, out_channels, kernel_size=1)
        self.elu = nn.ELU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        res = self.residual(x)
        out = self.elu(out + res)
        return out

class DRUnet(nn.Module):
    def __init__(self, input_channels=1, dims=32):
        super(DRUnet, self).__init__()
        
        self.conv1 = ConvLayer(input_channels, 16, kernel_size=1)
        self.block1_256 = Block1(16, dims)
        self.block3_256 = Block3(dims // 4, dims)
        self.maxpool_256 = nn.MaxPool2d(2, stride=2)

        self.block1_128 = Block1(dims, dims * 2)
        self.block3_128 = Block3(dims * 2 // 4, dims * 2)
        self.maxpool_128 = nn.MaxPool2d(2, stride=2)

        self.block1_64 = Block1(dims * 2, dims * 4)
        self.block3_64 = Block3(dims * 4 // 4, dims * 4)
        self.maxpool_64 = nn.MaxPool2d(2, stride=2)

        self.block1_32 = Block1(dims * 4, dims * 8)
        self.block3_32 = Block3(dims * 8 // 4, dims * 8)
        self.maxpool_32 = nn.MaxPool2d(2, stride=2)

        self.block1_16 = Block1(dims * 8, dims * 16)
        self.block3_16 = Block3(dims * 16 // 4, dims * 16)
        self.maxpool_16 = nn.MaxPool2d(2, stride=2)

        self.block1_8 = Block1(dims * 16, dims * 32)
        self.block3_8 = Block3(dims * 32 // 4, dims * 32)

        self.up_16 = nn.ConvTranspose2d(dims * 32, dims * 16, kernel_size=2, stride=2)
        self.batch_norm_1024 = nn.BatchNorm2d(dims * 16 * 2)
        self.elu_1024 = nn.ELU()

        self.block1_up16 = Block1(dims * 16 * 2, dims * 16)
        self.block3_up16 = Block3(dims * 16 // 4, dims * 16)


        self.up_32 = nn.ConvTranspose2d(dims * 16, dims * 8, kernel_size=2, stride=2)
        self.batch_norm_512 = nn.BatchNorm2d(dims * 8 * 2)
        self.elu_512 = nn.ELU()

        self.block1_up32 = Block1(dims * 8 * 2, dims * 8)
        self.block3_up32 = Block3(dims * 8 // 4, dims * 8)

        self.up_64 = nn.ConvTranspose2d(dims * 8, dims * 4, kernel_size=2, stride=2)
        self.batch_norm_256 = nn.BatchNorm2d(dims * 4 * 2)
        self.elu_256 = nn.ELU()

        self.block1_up64 = Block1(dims * 4 * 2, dims * 4)
        self.block3_up64 = Block3(dims * 4 // 4, dims * 4)

        self.up_128 = nn.ConvTranspose2d(dims * 4, dims * 2, kernel_size=2, stride=2)
        self.batch_norm_128 = nn.BatchNorm2d(dims * 2 * 2)
        self.elu_128 = nn.ELU()

        self.block1_up128 = Block1(dims * 2 * 2, dims * 2)
        self.block3_up128 = Block3(dims * 2 // 4, dims * 2)

        self.up_256 = nn.ConvTranspose2d(dims * 2, dims * 1, kernel_size=2, stride=2)
        self.batch_norm_64 = nn.BatchNorm2d(dims * 1 * 2)
        self.elu_64 = nn.ELU()

        self.block1_up256 = Block1(dims * 1 * 2, dims)
        self.block3_up256 = Block3(dims // 4, dims // 2)

        self.conv_out = nn.Conv2d(dims // 2, 1, kernel_size=1)

    def forward(self, x):

        out = self.conv1(x)

        out = self.block1_256(out)
        block3_256_out = self.block3_256(out)
        out = self.maxpool_256(block3_256_out)

        out = self.block1_128(out)
        block3_128_out = self.block3_128(out)
        out = self.maxpool_128(block3_128_out)

        out = self.block1_64(out)
        block3_64_out = self.block3_64(out)
        out = self.maxpool_64(block3_64_out)

        out = self.block1_32(out)
        block3_32_out = self.block3_32(out)
        out = self.maxpool_32(block3_32_out)

        out = self.block1_16(out)
        block3_16_out = self.block3_16(out)
        out = self.maxpool_16(block3_16_out)

        out = self.block1_8(out)
        out = self.block3_8(out)

        out = self.up_16(out)
        out = torch.cat((out, block3_16_out), dim=1)
        out = self.batch_norm_1024(out)
        out = self.elu_1024(out)
        out = self.block1_up16(out)
        out = self.block3_up16(out)

        out = self.up_32(out)
        out = torch.cat((out, block3_32_out), dim=1)
        out = self.batch_norm_512(out)
        out = self.elu_512(out)
        out = self.block1_up32(out)
        out = self.block3_up32(out)

        out = self.up_64(out)
        out = torch.cat((out, block3_64_out), dim=1)
        out = self.batch_norm_256(out)
        out = self.elu_256(out)
        out = self.block1_up64(out)
        out = self.block3_up64(out)

        out = self.up_128(out)
        out = torch.cat((out, block3_128_out), dim=1)
        out = self.batch_norm_128(out)
        out = self.elu_128(out)
        out = self.block1_up128(out)
        out = self.block3_up128(out)

        out = self.up_256(out)
        out = torch.cat((out, block3_256_out), dim=1)
        out = self.batch_norm_64(out)
        out = self.elu_64(out)
        out = self.block1_up256(out)
        out = self.block3_up256(out)

        out = self.conv_out(out)

        return out
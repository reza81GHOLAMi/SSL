import os

import torch
import torch.nn as nn
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Norm(nn.Module):
    def __init__(self, channels, num_features):
        super(Norm, self).__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels, affine=False, eps=1e-4)

        if num_features is not None:
            self.fcw = nn.Linear(num_features, channels)
            self.fcb = nn.Linear(num_features, channels)

    def forward(self, x, features=None):
        out = self.norm(x)
        if features is not None:
            w = self.fcw(features)
            b = self.fcb(features)
            out = w.view(-1, self.channels, 1, 1) * out + b.view(-1, self.channels, 1, 1)
        return out


class ResDown(nn.Module):
    """
    Residual down sampling block
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, num_features=512):
        super(ResDown, self).__init__()
        self.num_features = num_features
        self.norm1 = Norm(channel_in, num_features=num_features)
        self.norm2 = Norm(channel_out, num_features=num_features)
        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size, 2, kernel_size // 2)
        self.conv2 = nn.Conv2d(channel_out, channel_out, kernel_size, 1, kernel_size // 2)
        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 2, kernel_size // 2)
        self.act_fnc = nn.GELU()
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x, features=None):
        x = self.act_fnc(self.norm1(x, features))
        skip = self.conv3(x)
        x = self.conv1(x)
        x = self.norm2(x, features)
        x = self.act_fnc(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x + skip


class ResBlock(nn.Module):
    """
    Residual block
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, num_features=128):
        super(ResBlock, self).__init__()
        self.norm1 = Norm(channel_in, num_features=num_features)
        self.conv1 = nn.Conv2d(channel_in, channel_in, kernel_size, 1, kernel_size // 2)
        self.norm2 = Norm(channel_in, num_features=num_features)
        self.conv2 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)

        if not channel_in == channel_out:
            self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)
        else:
            self.conv3 = None

        self.act_fnc = nn.GELU()
        self.drop = nn.Dropout2d(0.05)

    def forward(self, x_in, features=None):
        x = self.act_fnc(self.norm1(x_in, features))
        if self.conv3 is not None:
            skip = self.conv3(x)
        else:
            skip = x_in
        x = self.conv1(x)
        x = self.norm2(x, features)
        x = self.act_fnc(x)
        x = self.conv2(x)
        x = self.drop(x)

        return x + skip


class Encoder(nn.Module):
    """
    Encoder block
    """

    def __init__(self, channels, ch=64, blocks=(1, 2, 4), num_features=128):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv2d(channels, blocks[0] * ch, 3, 1, 1)

        widths_in = list(blocks)
        widths_out = list(blocks[1:]) + [blocks[-1]]

        self.layer_blocks = nn.ModuleList([])

        for w_in, w_out in zip(widths_in, widths_out):
            self.layer_blocks.append(ResDown(w_in * ch, w_out * ch, num_features=num_features))

        self.block_out = ResBlock(w_out * ch, w_out * ch, num_features=num_features)

        self.act_fnc = nn.GELU()

    def forward(self, x, index_features=None):
        x = self.conv_in(x)

        for block in self.layer_blocks:
            x = block(x, index_features)

        x = self.block_out(x, index_features)
        return x


class Classifier(nn.Module):
    def __init__(self, channel_in=8, num_classes=6, attr_feat_in=2, attr_feat_out=None):
        super(Classifier, self).__init__()
        self.encoder = Encoder(channel_in, ch=8, blocks=(4, 8, 16, 24, 32, 64), num_features=attr_feat_out)

        last_conv = self.encoder.layer_blocks[-1].conv2

        self.fc1 = nn.Linear(last_conv.out_channels,
                             int(last_conv.out_channels / 2))
        self.fc2 = nn.Linear(int(last_conv.out_channels / 2),
                             num_classes)
        self.act = nn.GELU()
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        features = self.encoder(x)
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.shape[0], -1)
        out = self.fc1(pooled)
        out = self.act(out)
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    x = torch.rand([1, 8, 256, 256]).to(torch.device("cuda:0"))

    u_net = Classifier(channel_in=8).to(torch.device("cuda:0"))

    with torch.no_grad():
        out = u_net(x)
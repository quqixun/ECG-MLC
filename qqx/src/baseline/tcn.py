import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        return

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size,
                 stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs,
                      kernel_size, stride=stride,
                      padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs,
                      kernel_size, stride=stride,
                      padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = None
        if n_inputs != n_outputs:
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1)
        self.relu = nn.ReLU()
        self.init_weights()
        return

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
        return

    def forward(self, x):
        out = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(out + x)


class TemporalConvNet(nn.Module):
    def __init__(self, in_channels, num_channels,
                 kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            in_channels = in_channels if i == 0 else num_channels[i - 1]
            layers += [
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=padding, dropout=dropout
                )
            ]
        self.network = nn.Sequential(*layers)
        return

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, in_channels, out_channels,
                 num_channels, kernel_size, dropout=0.2):
        super(TCN, self).__init__()

        self.tcn = TemporalConvNet(
            in_channels, num_channels,
            kernel_size=kernel_size, dropout=dropout
        )
        self.linear = nn.Linear(num_channels[-1], out_channels)
        return

    def forward(self, x):
        out = self.tcn(x)
        out = self.linear(out[:, :, -1])
        return out


class MSTCN(nn.Module):
    def __init__(self, in_channels, out_channels,
                 num_channels, kernel_size, dropout=0.2):
        super(MSTCN, self).__init__()

        self.tcn = TemporalConvNet(
            in_channels, num_channels,
            kernel_size=kernel_size, dropout=dropout
        )
        self.tcn0 = self.tcn.network[0]
        self.tcn1 = self.tcn.network[1]
        self.tcn2 = self.tcn.network[2]
        self.tcn3 = self.tcn.network[3]
        self.tcn4 = self.tcn.network[4]
        self.tcn5 = self.tcn.network[5]
        self.linear = nn.Linear(sum(num_channels), out_channels)
        return

    def forward(self, x):
        out0 = self.tcn0(x)
        out1 = self.tcn1(out0)
        out2 = self.tcn2(out1)
        out3 = self.tcn3(out2)
        out4 = self.tcn4(out3)
        out5 = self.tcn5(out4)
        out = torch.cat([out0[:, :, -1], out1[:, :, -1],
                         out2[:, :, -1], out3[:, :, -1],
                         out4[:, :, -1], out5[:, :, -1]], dim=1)
        out = self.linear(out)
        return out


if __name__ == '__main__':
    from torchsummary import summary

    # model = TCN(8, 55, [32, 32, 64, 64, 128, 128], 7, 0.2)
    model = MSTCN(8, 55, [32, 32, 64, 64, 128, 128], 7, 0.2)
    summary(model, input_size=(8, 1000), batch_size=128)

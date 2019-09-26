import torch.nn as nn


class Res1DBlock(nn.Module):

    def __init__(self, inplanes, planes, inp=16, ks=3, stride=1,
                 groups=1, expansion=4):
        super(Res1DBlock, self).__init__()

        if groups == 1:
            D = planes
        else:
            D = groups * int(planes / expansion)

        self.conv1 = self.__conv1(inplanes, D)
        self.bn1 = nn.BatchNorm1d(D)

        self.conv2 = self.__convn(D, D, ks, stride, groups)
        self.bn2 = nn.BatchNorm1d(D)

        self.conv3 = self.__conv1(D, planes * expansion)
        self.bn3 = nn.BatchNorm1d(planes * expansion)

        # self.active = nn.ReLU(inplace=True)
        self.active = nn.LeakyReLU(0.1, inplace=True)
        self.shortcut = None
        if stride != 1 or inplanes != planes * expansion:
            self.shortcut = nn.Sequential(
                self.__conv1(inplanes, planes * expansion, stride),
                nn.BatchNorm1d(planes * expansion)
            )
        return

    def __conv1(self, inplanes, outplanes, stride=1):
        return nn.Conv1d(inplanes, outplanes, kernel_size=1,
                         stride=stride, padding=0, bias=True)

    def __convn(self, inplanes, outplanes, ks=3, stride=1, groups=1):
        return nn.Conv1d(inplanes, outplanes, kernel_size=ks,
                         stride=stride, groups=groups, bias=True,
                         padding=int((ks - 1) / 2))

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.active(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.active(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = self.active(out)
        return out


class ResNeXt(nn.Module):

    def __init__(self, in_channels=12, n_classes=55, kss=[15, 15, 11, 11, 7],
                 layers=[3, 3, 3, 3, 3], groups=1, dropout=0.5):
        super(ResNeXt, self).__init__()

        self.inp = 32
        self.expa = 4
        self.groups = groups

        self.iconv = nn.Conv1d(in_channels, self.inp, 15, padding=7, bias=False)
        self.ibn = nn.BatchNorm1d(self.inp)
        self.mbranch = self.__make_branch(layers, kss)

        feats = self.inp * self.expa * 16
        self.linear = nn.Linear(feats, n_classes)

        self.dropout = nn.Dropout(dropout)
        # self.active = nn.ReLU(inplace=True)
        self.active = nn.LeakyReLU(0.1, inplace=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return

    def __make_branch(self, layers, kss):
        self.inplanes = self.inp
        branch_layers = [
            self.__make_layers(self.inp * 1, layers[0], kss[0], 1),
            self.__make_layers(self.inp * 2, layers[1], kss[1], 2),
            self.__make_layers(self.inp * 4, layers[2], kss[2], 2),
            self.__make_layers(self.inp * 8, layers[3], kss[3], 2),
            self.__make_layers(self.inp * 16, layers[4], kss[4], 2)
        ]

        return nn.Sequential(*branch_layers)

    def __make_layers(self, planes, blocks, ks=3, stride=1):
        layers = [
            Res1DBlock(self.inplanes, planes, self.inp,
                       ks, stride, self.groups, self.expa)
        ]
        self.inplanes = planes * self.expa
        for _ in range(1, blocks):
            layers.append(
                Res1DBlock(self.inplanes, planes, self.inp,
                           ks, 1, self.groups, self.expa)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.iconv(x)
        out = self.ibn(out)
        out = self.active(out)
        out = self.maxpool(out)
        out = self.mbranch(out)
        out = self.avgpool(out)

        out = out.view((-1, out.size()[1]))
        out = self.dropout(out)
        out = self.linear(out)
        return out


if __name__ == '__main__':
    from torchsummary import summary

    model = ResNeXt(kss=[15, 15, 11, 11, 7], layers=[3, 3, 3, 3, 3], groups=8)
    summary(model, (12, 1000), batch_size=256)

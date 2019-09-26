import torch
import pickle
import numpy as np
import torch.nn as nn

from torch.nn import Parameter


class GraphConvBlock(nn.Module):

    def __init__(self, inplanes, planes, bias=False):
        super(GraphConvBlock, self).__init__()

        self.weight = Parameter(torch.Tensor(inplanes, planes))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, planes))
        else:
            self.register_parameter('bias', None)
        return

    def forward(self, emb, adj):
        out = torch.matmul(emb, self.weight)
        out = torch.matmul(adj, out)
        if self.bias is not None:
            return out + self.bias
        else:
            return out


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

        self.active = nn.LeakyReLU(0.2, inplace=True)
        self.shortcut = None
        if stride != 1 or inplanes != planes * expansion:
            self.shortcut = nn.Sequential(
                self.__conv1(inplanes, planes * expansion, stride),
                nn.BatchNorm1d(planes * expansion)
            )
        return

    def __conv1(self, inplanes, outplanes, stride=1):
        return nn.Conv1d(inplanes, outplanes, kernel_size=1,
                         stride=stride, padding=0, bias=False)

    def __convn(self, inplanes, outplanes, ks=3, stride=1, groups=1):
        return nn.Conv1d(inplanes, outplanes, kernel_size=ks,
                         stride=stride, groups=groups, bias=False,
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


class GraphResNeXt(nn.Module):

    def __init__(self, in_channels=12, n_classes=55, kss=[13, 11, 9, 7, 5],
                 layers=[3, 3, 3, 3, 3], groups=8, dropout=0.5,
                 hrv_feats=None, graph_pkl=None):
        super(GraphResNeXt, self).__init__()

        self.inp = 32
        self.expa = 4
        self.groups = groups

        self.iconv = nn.Conv1d(in_channels, self.inp, 15, padding=7, bias=False)
        self.ibn = nn.BatchNorm1d(self.inp)
        self.mbranch = self.__make_branch(layers, kss)

        self.dropout = nn.Dropout(dropout)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.active = nn.LeakyReLU(0.2, inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        feats = self.inp * self.expa * 16
        self.linear = nn.Linear(feats, n_classes, bias=True)

        if hrv_feats is not None:
            self.use_hrv = True
            self.tanh = nn.Tanh()
            self.hrv_linear = nn.Linear(hrv_feats, 512, bias=True)
            feats += 512
            self.linear = nn.Linear(feats, n_classes, bias=True)

        if graph_pkl is not None:
            self.use_graph = True
            graph_dict = pickle.load(open(graph_pkl, 'rb'))
            self.relation = Parameter(torch.from_numpy(
                self.__relation_matrix(graph_dict, 0.1)
            ).float())
            self.graph = self.__init_graph()
            emb = torch.tensor(graph_dict['emb'])
            self.emb = torch.autograd.Variable(emb).float().detach()
            if torch.cuda.is_available():
                self.emb = self.emb.cuda()
                self.graph = self.graph.cuda()
            self.gc1 = GraphConvBlock(emb.size()[1], 1024)
            self.gc2 = GraphConvBlock(1024, feats)

        self.__init_parameters()
        return

    def __init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, GraphConvBlock)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return

    def __make_branch(self, layers, kss):
        self.inplanes = self.inp
        branch_layers = [
            self.__make_layers(self.inp * 1, layers[0], kss[0], 2),
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

    def __relation_matrix(self, graph_dict, threshold=0.1):
        relation = graph_dict['matrix'] / graph_dict['nums']
        relation[relation < threshold] = 0
        relation[relation >= threshold] = 1

        relation = relation / (relation.sum(0, keepdims=True) + 1e-6)
        relation = relation + np.identity(relation.shape[0], np.int)
        return relation

    def __init_graph(self):
        graph = torch.pow(self.relation.sum(1).float(), -0.5)
        graph = torch.diag(graph)
        graph = torch.matmul(torch.matmul(self.relation, graph).t(), graph)
        return graph.detach()

    def forward(self, ecg, hrv=None):
        out = self.iconv(ecg)
        out = self.ibn(out)
        out = self.active(out)
        out = self.maxpool(out)
        out = self.mbranch(out)
        out = self.avgpool(out)
        feats = out.view((-1, out.size()[1]))

        if hrv is not None and self.use_hrv:
            if len(hrv.size()) == 3:
                hrv = hrv.view((-1, hrv.size()[2]))
            hrv = self.dropout(hrv)
            hrv = self.hrv_linear(hrv)
            hrv = self.tanh(hrv)
            feats = torch.cat([feats, hrv], dim=1)

        if self.use_graph:
            gc = self.gc1(self.emb, self.graph)
            gc = self.active(gc)
            gc = self.gc2(gc, self.graph)
            gc = gc.transpose(0, 1)
            pred = torch.matmul(feats, gc)
        else:
            feats = self.dropout(feats)
            pred = self.linear(feats)

        return pred


if __name__ == '__main__':
    # from torchsummary import summary

    # model = ResNeXt(
    #     kss=[13, 11, 9, 7, 5],
    #     layers=[3, 3, 3, 3, 3],
    #     groups=8, hrv_feats=2212
    # )
    model = GraphResNeXt(
        kss=[3, 3, 3, 3, 3],
        layers=[3, 3, 3, 3, 3],
        groups=1, hrv_feats=2212,
        graph_pkl='./graph.pkl'
    )

    ecg = torch.rand(16, 12, 2000)
    hrv = torch.rand(16, 2212)
    emb = torch.rand(55, 300)
    pred = model(ecg, hrv)
    print(pred.size())
    # summary(model, [(12, 2000), (1, 2212), (55, 200)], batch_size=256)

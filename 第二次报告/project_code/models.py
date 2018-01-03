import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable


# feature alignment net, bidirectional GAN used
class FeatAggregate(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(FeatAggregate, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # bidirectional GRN
        self.GRU = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, dropout=dropout, bidirectional=True)

        self.init_params()

    # initialize parameters
    def init_params(self, bias=1.0):
        # GRN initialization(orthogonal and constant initialisubzation)
        for name, param in self.GRU.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, bias)
            elif 'weight' in name:
                # init.orthogonal(param)
                init.xavier_uniform(param)

    def forward(self, feats):
        # state initialization
        h0 = Variable(torch.zeros(self.num_layers * 2, feats.size(1), self.hidden_size).float())

        if feats.is_cuda:
            h0 = h0.cuda()

        output, _ = self.GRU(feats, h0)

        return output


class MultiModal(nn.Module):
    def __init__(self, num_layers=2, kernel_size=3, dropout=0.5, hidden_size=None):
        super(MultiModal, self).__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        assert (kernel_size % 2 == 1)
        self.dropout = dropout
        if hidden_size is None:
            self.hidden_size = [128] * num_layers
        else:
            self.hidden_size = hidden_size
        assert (len(hidden_size) == num_layers)

        self.v_fc, self.a_fc, self.rnn_fc = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                for j in range(kernel_size):
                    self.v_fc.append(nn.Linear(in_features=1024, out_features=hidden_size[0]))
                    self.a_fc.append(nn.Linear(in_features=128, out_features=hidden_size[0]))
                self.rnn_fc.append(nn.Linear(in_features=256 * 2, out_features=hidden_size[0]))
            else:
                for j in range(kernel_size):
                    self.v_fc.append(nn.Linear(in_features=hidden_size[i - 1], out_features=hidden_size[i]))
                    self.a_fc.append(nn.Linear(in_features=hidden_size[i - 1], out_features=hidden_size[i]))
                self.rnn_fc.append(nn.Linear(in_features=hidden_size[i - 1], out_features=hidden_size[i]))

        self.tanh = nn.Tanh()
        self.do = nn.Dropout(p=dropout)
        self.a_featpool = FeatAggregate(input_size=128, hidden_size=256, num_layers=3, dropout=dropout)

        self.init_params()

    def init_params(self):
        self.a_featpool.init_params()

        for i in range(len(self.v_fc)):
            init.xavier_normal(self.v_fc[i].weight)
            init.constant(self.v_fc[i].bias, 0.1)

            init.xavier_normal(self.a_fc[i].weight)
            init.constant(self.a_fc[i].bias, 0.1)

        for i in range(self.num_layers):
            init.xavier_normal(self.rnn_fc[i].weight)
            init.constant(self.rnn_fc[i].bias, 0.1)

    def forward(self, vfeat, afeat):
        batch_size = afeat.size(0)
        is_cuda = afeat.is_cuda
        padding = self.kernel_size // 2

        vfeat_last, afeat_last = vfeat, afeat
        for i in range(self.num_layers):
            if i == 0:
                # layer1
                v_zero_pad = Variable(torch.zeros(batch_size, 1, 1024))
                v_zero_pad = v_zero_pad.cuda() if is_cuda else v_zero_pad
                a_zero_pad = Variable(torch.zeros(batch_size, 1, 128))
                a_zero_pad = a_zero_pad.cuda() if is_cuda else a_zero_pad

                # padding
                vfeat_cur = [v_zero_pad] * padding + [vfeat_last] + [v_zero_pad] * padding
                vfeat_cur = self.do(torch.cat(vfeat_cur, dim=1))
                afeat_cur = [a_zero_pad] * padding + [afeat_last] + [a_zero_pad] * padding
                afeat_cur = self.do(torch.cat(afeat_cur, dim=1))

                # time local fc
                vfeat_last, afeat_last = 0, 0
                for j in range(self.kernel_size):
                    v = self.v_fc[j](vfeat_cur)
                    a = self.a_fc[j](afeat_cur)
                    start, end = j, j - self.kernel_size + 1
                    if end != 0:
                        vfeat_last = vfeat_last + v[:, start:end, :]
                        afeat_last = afeat_last + a[:, start:end, :]
                    else:
                        vfeat_last = vfeat_last + v[:, start:, :]
                        afeat_last = afeat_last + a[:, start:, :]
                if i < self.num_layers - 1:
                    vfeat_last = 1.7159 * self.tanh(0.6666667 * vfeat_last)
                    afeat_last = 1.7159 * self.tanh(0.6666667 * afeat_last)

            else:
                v_zero_pad = Variable(torch.zeros(batch_size, 1, self.hidden_size[i - 1]))
                v_zero_pad = v_zero_pad.cuda() if is_cuda else v_zero_pad
                a_zero_pad = Variable(torch.zeros(batch_size, 1, self.hidden_size[i - 1]))
                a_zero_pad = a_zero_pad.cuda() if is_cuda else a_zero_pad

                # padding
                vfeat_cur = [v_zero_pad] * padding + [vfeat_last] + [v_zero_pad] * padding
                vfeat_cur = self.do(torch.cat(vfeat_cur, dim=1))
                afeat_cur = [a_zero_pad] * padding + [afeat_last] + [a_zero_pad] * padding
                afeat_cur = self.do(torch.cat(afeat_cur, dim=1))

                # time local fc
                vfeat_last, afeat_last = 0, 0
                for j in range(self.kernel_size):
                    v = self.v_fc[i*self.kernel_size + j](vfeat_cur)
                    a = self.a_fc[i*self.kernel_size + j](afeat_cur)
                    start, end = j, j - self.kernel_size + 1
                    if end != 0:
                        vfeat_last = vfeat_last + v[:, start:end, :]
                        afeat_last = afeat_last + a[:, start:end, :]
                    else:
                        vfeat_last = vfeat_last + v[:, start:, :]
                        afeat_last = afeat_last + a[:, start:, :]
                if i < self.num_layers - 1:
                    vfeat_last = 1.7159 * self.tanh(0.6666667 * vfeat_last)
                    afeat_last = 1.7159 * self.tanh(0.6666667 * afeat_last)

        rnn_a = self.a_featpool.forward(afeat.transpose(0, 1))

        rnn_fc = 0
        for i in range(self.num_layers):
            if i == 0:
                rnn_fc = self.rnn_fc[0](self.do(rnn_a.transpose(0, 1)))
                if i < self.num_layers - 1:
                    rnn_fc = 1.7159 * self.tanh(0.6666667 * rnn_fc)
            else:
                rnn_fc = self.rnn_fc[i](rnn_fc)
                if i < self.num_layers - 1:
                    rnn_fc = 1.7159 * self.tanh(0.6666667 * rnn_fc)

        return 1.7159 * self.tanh(0.6666667 * (rnn_fc + afeat_last + vfeat_last))


class VAMetric(nn.Module):
    def __init__(self):
        super(VAMetric, self).__init__()

        self.model = MultiModal(num_layers=2, kernel_size=3, dropout=0.3, hidden_size=[384, 384])
        self.fc = nn.Linear(in_features=384, out_features=2)
        init.xavier_normal(self.fc.weight)
        init.constant(self.fc.bias, 0.1)

        self.do = nn.Dropout(p=0.3)
        self.maxpool = nn.MaxPool1d(kernel_size=5)

        self.softmax = nn.Softmax()

    def forward(self, vfeat, afeat, stage='TEST'):
        if stage == 'TEST':
            vfeat.volatile = True
            afeat.volatile = True

        batch_size = vfeat.size(0)
        feat = self.model.forward(vfeat=vfeat, afeat=afeat)
        feat = self.fc(self.do(feat))

        if stage == 'TEST':
            feat_chunks = [self.softmax(feat[:, i, :])[:, 1].resize(batch_size, 1, 1) for i in range(feat.size(1))]
            results = self.maxpool(torch.cat(feat_chunks, dim=2))
            results = 1 - torch.mean(results.view(batch_size, -1), dim=1)
            return results
        else:
            feat_chunks = [self.softmax(feat[:, i, :])[:, 1].resize(batch_size, 1, 1) for i in range(feat.size(1))]
            results = self.maxpool(torch.cat(feat_chunks, dim=2))
            results = [results[:, :, i] for i in range(results.size(2))]
            return results


# focal loss
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, eps=1e-7):
        super(ContrastiveLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, dist, label):
        loss = Variable(torch.zeros(label.size(0)).float(), requires_grad=True)
        loss = loss.cuda() if label.is_cuda else loss

        mask = torch.zeros(label.size(0), 2).scatter_(1, label.cpu().data.unsqueeze(dim=1), 1).byte()
        mask = Variable(mask)
        mask = mask.cuda() if label.is_cuda else mask

        # focal loss
        for i in range(len(dist)):
            pt = torch.masked_select(torch.cat([1 - dist[i], dist[i]], dim=1), mask)
            pt = pt.clamp(self.eps, 1.0 - self.eps)
            loss = loss - torch.pow(1 - pt, self.gamma) * torch.log(pt)
        return loss.mean()

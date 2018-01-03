from __future__ import print_function
import numpy as np
from easydict import EasyDict
from BoostedForest import BoostedForest
from sklearn.preprocessing import StandardScaler

import os
from optparse import OptionParser
import extract_models
from dataset import VideoFeatDataset as dset
from tools.config_tools import Config

import torch
from torch.autograd import Variable

# load opts
parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="training configuration",
                  default="./configs/bf_train_config.yaml")

(bf_opts, args) = parser.parse_args()
assert isinstance(bf_opts, object)
bf_opt = Config(bf_opts.config)
print(bf_opt)


def extract_feats(data_loader, deep_model):
    # inferrance
    deep_model.eval()

    X, y = [], []

    cum_sample = 0
    num_sample = len(data_loader)
    for vfeat0, afeat0 in data_loader:
        # shuffling the index orders
        bz = vfeat0.size()[0]
        orders = np.arange(bz).astype('int32')
        shuffle_orders1 = orders.copy()
        np.random.shuffle(shuffle_orders1)
        shuffle_orders2 = orders.copy()
        np.random.shuffle(shuffle_orders2)
        shuffle_orders3 = orders.copy()
        np.random.shuffle(shuffle_orders3)

        # creating a new data with the shuffled indices
        afeat1 = afeat0[torch.from_numpy(shuffle_orders1).long()].clone()
        afeat2 = afeat0[torch.from_numpy(shuffle_orders2).long()].clone()
        afeat3 = afeat0[torch.from_numpy(shuffle_orders3).long()].clone()
        vfeat1 = vfeat0
        vfeat2 = vfeat0
        vfeat3 = vfeat0

        afeat = torch.cat((afeat0, afeat1, afeat2, afeat3), 0)
        vfeat = torch.cat((vfeat0, vfeat1, vfeat2, vfeat3), 0)

        # generating the labels
        # 1. the labels for the shuffled feats
        label1 = (orders == shuffle_orders1 + 0).astype('float32')
        label2 = (orders == shuffle_orders2 + 0).astype('float32')
        label3 = (orders == shuffle_orders3 + 0).astype('float32')

        # 2. the labels for the original feats
        label0 = label1.copy()
        label0[:] = 1

        # concat the labels together
        target = np.concatenate((label0, label1, label2, label3))
        y.append(target)

        # transpose the feats
        vfeat = vfeat.transpose(2, 1)
        afeat = afeat.transpose(2, 1)

        # put the data into Variable
        vfeat_var = Variable(vfeat)
        afeat_var = Variable(afeat)

        # if you have gpu, then shift data to GPU
        if bf_opt.cuda:
            vfeat_var = vfeat_var.cuda()
            afeat_var = afeat_var.cuda()

        # forward, backward optimize
        feat = deep_model.forward(vfeat_var, afeat_var)
        X.append(feat.cpu().data.numpy())

        cum_sample += 1
        print('extract deep features: {} / {}'.format(cum_sample, num_sample))

    X = np.concatenate(X)
    y = np.concatenate(y)
    print('extract deep features complete.')
    return X, y


if __name__ == '__main__':
    # load dataset
    train_dataset = dset(bf_opt.data_dir, flist=bf_opt.flist)
    print('number of train samples is: {0}'.format(len(train_dataset)))
    print('finished loading data')

    # train data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bf_opt.batchSize,
                                               shuffle=True, num_workers=int(bf_opt.workers))

    # create neural network deep_model
    deep_model = extract_models.VAMetric()

    if bf_opt.init_deep_model != '':
        print('loading pretrained deep_model from {0}'.format(bf_opt.init_deep_model))
        if bf_opt.cuda:
            deep_model.load_state_dict(torch.load(bf_opt.init_deep_model, map_location=lambda storage, loc: storage.cuda(int(bf_opt.gpu_id) - 1)))
        else:
            deep_model.load_state_dict(torch.load(bf_opt.init_deep_model, map_location=lambda storage, loc: storage))

    if bf_opt.cuda:
        print('shift deep_model and criterion to GPU .. ')
        deep_model = deep_model.cuda()

    data, target = extract_feats(train_loader, deep_model)
    # balance pos and neg weights
    samp_weight = np.ones(target.shape, dtype=float)
    samp_weight[target == 1] /= sum(target == 1)
    samp_weight[target == 0] /= sum(target == 0)
    samp_weight /= sum(samp_weight)

    # boosted tree config
    opts = EasyDict()
    opts.nWeak = [16, 64, 128, 256, 512]
    opts.verbose = True
    opts.pTree = EasyDict()
    opts.pTree.max_depth = 3            # decision tree max depth
    opts.pTree.frac_feat = 'sqrt'       # choose sqrt(num_feat) each split
    opts.pTree.frac_samp = 0.3          # choose 50% samples each split

    bf = BoostedForest(opts)
    bf.train(data, target, sample_weight=samp_weight, save_path='./checkpoints/')
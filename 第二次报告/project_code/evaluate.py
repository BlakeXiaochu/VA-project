#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

import os
import numpy as np
from optparse import OptionParser

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import models
from dataset import VideoFeatDataset as dset
from tools.config_tools import Config
from tools import utils
import time


parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="evaluation configuration",
                  default="./configs/test_config.yaml")

(test_opts, args) = parser.parse_args()
assert isinstance(test_opts, object)
test_opt = Config(test_opts.config)
print(test_opt)

if test_opt.checkpoint_folder is None:
    test_opt.checkpoint_folder = 'checkpoints'

test_video_dataset = dset(test_opt.data_dir, test_opt.video_flist, which_feat='vfeat')
test_audio_dataset = dset(test_opt.data_dir, test_opt.audio_flist, which_feat='afeat')

print('number of test samples is: {0}'.format(len(test_video_dataset)))
print('finished loading data')


if torch.cuda.is_available() and not test_opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with \"cuda: True\"")
else:
    if int(test_opt.ngpu) == 1:
        print('so we use gpu 1 for testing')
        os.environ['CUDA_VISIBLE_DEVICES'] = test_opt.gpu_id
        cudnn.benchmark = True
        print('setting gpu on gpuid {0}'.format(test_opt.gpu_id))


# test function for metric learning
def test(video_loader, audio_loader, model, test_opt):
    # inference mode
    model1, model2 = model
    model1.eval()
    model2.eval()

    sim_mat = []
    right = 0
    for _, vfeat in enumerate(video_loader):
        for _, afeat in enumerate(audio_loader):
            bz = vfeat.size()[0]
            simmat = 0
            for k in np.arange(bz):
                cur_vfeat = vfeat[k].clone()
                cur_vfeats = cur_vfeat.repeat(bz, 1, 1)

                vfeat_var = Variable(cur_vfeats)
                afeat_var = Variable(afeat)

                vfeat_var.volatile = True
                afeat_var.volatile = True

                if test_opt.cuda:
                    vfeat_var = vfeat_var.cuda()
                    afeat_var = afeat_var.cuda()

                st = time.time()
                cur_sim1 = model1.forward(vfeat_var, afeat_var, stage='TEST')
                cur_sim2 = model2.forward(vfeat_var, afeat_var, stage='TEST')
                cur_sim = (cur_sim1 + cur_sim2) / 2.0
                et = time.time()

                cur_sim = cur_sim.view(cur_sim.size(0), -1)
                if k == 0:
                    simmat = cur_sim.clone()
                else:
                    simmat = torch.cat((simmat, cur_sim), 1)
            sorted, indices = torch.sort(simmat, 0)
            np_indices = indices.cpu().data.numpy()
            topk = np_indices[:test_opt.topk,:]
            for k in np.arange(bz):
                order = topk[:,k]
                if k in order:
                    right = right + 1
            print('The similarity matrix: \n {}'.format(simmat))
            print('Cost time: {}, Testing accuracy (top{}): {:.3f}'.format(et - st, test_opt.topk, right/bz))


def main():
    global test_opt
    # test data loader
    test_video_loader = torch.utils.data.DataLoader(test_video_dataset, batch_size=test_opt.batchSize,
                                     shuffle=False, num_workers=int(test_opt.workers))
    test_audio_loader = torch.utils.data.DataLoader(test_audio_dataset, batch_size=test_opt.batchSize,
                                     shuffle=False, num_workers=int(test_opt.workers))

    # create model
    model1 = models.VAMetric()
    model2 = models.VAMetric()

    if test_opt.init_model1 != '':
        print('loading pretrained model from {0}'.format(test_opt.init_model1))
        print('loading pretrained model from {0}'.format(test_opt.init_model2))
        if test_opt.cuda:
            model1.load_state_dict(torch.load(test_opt.init_model1, map_location=lambda storage, loc: storage.cuda(int(test_opt.gpu_id) - 1)))
            model2.load_state_dict(torch.load(test_opt.init_model2, map_location=lambda storage, loc: storage.cuda(int(test_opt.gpu_id) - 1)))
        else:
            model1.load_state_dict(torch.load(test_opt.init_model1, map_location=lambda storage, loc: storage))
            model2.load_state_dict(torch.load(test_opt.init_model2, map_location=lambda storage, loc: storage))
    if test_opt.cuda:
        print('shift model to GPU .. ')
        model1 = model1.cuda()
        model2 = model2.cuda()

    model = [model1, model2]
    test(test_video_loader, test_audio_loader, model, test_opt)


if __name__ == '__main__':
    main()

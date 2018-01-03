from __future__ import print_function
import numpy as np
import os
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

from optparse import OptionParser

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import extract_models
from dataset import VideoFeatDataset as dset
from tools.config_tools import Config


parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="evaluation configuration",
                  default="./configs/bf_test_config.yaml")

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
def test(video_loader, audio_loader, deep_model, bf, test_opt):
    # inference mode
    deep_model.eval()

    sim_mat = []
    right = [0] * test_opt.topk
    for _, vfeat in enumerate(video_loader):
        for _, afeat in enumerate(audio_loader):
            # transpose feats
            vfeat = vfeat.transpose(2, 1)
            afeat = afeat.transpose(2, 1)

            # shuffling the index orders
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

                feat = deep_model.forward(vfeat_var, afeat_var)
                feat = feat.cpu().data.numpy()
                # cur_sim = - bf.decision_function(feat)
                cur_sim = bf.predict_proba(feat)
                cur_sim = torch.from_numpy(cur_sim[:, 0].copy()).view(-1, 1)

                if k == 0:
                    simmat = cur_sim.clone()
                else:
                    simmat = torch.cat((simmat, cur_sim), 1)
            sorted, indices = torch.sort(simmat, 0)
            np_indices = indices.cpu().numpy()
            topk = np_indices[:test_opt.topk, :]
            for k in np.arange(bz):
                order = topk[:,k]
                for j in range(test_opt.topk):
                    if k in order[:j+1]:
                        right[j] = right[j] + 1
            print('The similarity matrix: \n {}'.format(simmat))
            right = [i/bz for i in right]
            print('Testing accuracy (top{}): {}'.format(test_opt.topk, right))


if __name__ == '__main__':
    # global test_opt
    # test data loader
    test_video_loader = torch.utils.data.DataLoader(test_video_dataset, batch_size=test_opt.batchSize,
                                                    shuffle=False, num_workers=int(test_opt.workers))
    test_audio_loader = torch.utils.data.DataLoader(test_audio_dataset, batch_size=test_opt.batchSize,
                                                    shuffle=False, num_workers=int(test_opt.workers))

    # load model
    print('loading pretrained bf model from {0}'.format(test_opt.init_bf_model))
    bf = joblib.load(test_opt.init_bf_model)

    deep_model = extract_models.VAMetric()
    if test_opt.init_deep_model != '':
        print('loading pretrained deep model from {0}'.format(test_opt.init_deep_model))
        if test_opt.cuda:
            deep_model.load_state_dict(torch.load(test_opt.init_deep_model, map_location=lambda storage, loc: storage.cuda(int(test_opt.gpu_id) - 1)))
        else:
            deep_model.load_state_dict(torch.load(test_opt.init_deep_model, map_location=lambda storage, loc: storage))
    if test_opt.cuda:
        print('shift model to GPU .. ')
        deep_model = deep_model.cuda()

    test(test_video_loader, test_audio_loader, deep_model, bf, test_opt)




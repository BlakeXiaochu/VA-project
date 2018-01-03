from __future__ import print_function
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from easydict import EasyDict
import time


class BoostedForest(object):
    __slots__ = {'opts', '_base_estimator', 'detector', '_stage'}

    def __init__(self, opts=EasyDict()):
        self.opts = opts
        self._init_opts()
        # self._base_estimator = DecisionTreeClassifier(criterion='entropy',
                                                      # max_depth=self.opts.pTree.max_depth,
                                                      # min_weight_fraction_leaf=self.opts.pTree.min_weight,
                                                      # max_features=self.opts.pTree.frac_feat)
        self.detector = None
        self._stage = 0

    # initialize forest training with default options
    def _init_opts(self):
        """
        pTree      - ['REQ'] parameters for binaryTreeTrain
        nWeak      - [128] number of trees to learn
        discrete   - [1] train Discrete-AdaBoost or Real-AdaBoost
        verbose    - [0] if true print status information
        """
        # pTree
        pTree = self.opts.get('pTree')
        if pTree is None: self.opts.pTree = EasyDict()

        max_depth = self.opts.pTree.get('max_depth')
        if max_depth is None: self.opts.pTree.max_depth = 3

        min_weight = self.opts.pTree.get('min_weight')
        if min_weight is None: self.opts.pTree.min_weight = 0.01

        frac_feat = self.opts.pTree.get('frac_feat')
        if frac_feat is None: self.opts.pTree.frac_feat = 1.0

        frac_samp = self.opts.pTree.get('frac_samp')
        if frac_samp is None: self.opts.pTree.frac_samp = 1.0

        nWeak = self.opts.get('nWeak')
        if nWeak is None: self.opts.nWeak = [128, ]

        discrete = self.opts.get('discrete')
        if discrete is None: self.opts.discrete = False

        verbose = self.opts.get('verbose')
        if verbose is None: self.opts.verbose = False

    # train boosted tree for one stage
    def train_one_stage(self, samples, targets, sample_weight=None, save=True, save_path='./'):
        """
        train the boosted tree

        :param samples: numpy.ndarray of shape = [n_samples, n_features]
                    The training input samples.
        :param targets: numpy.ndarray of shape = [n_samples]
                    The target values (class labels).
        :param sample_weight: numpy.ndarray of shape = [n_samples], optional
                    Sample weights. If None, the sample weights are initialized to 1 / n_samples
        """
        assert isinstance(samples, np.ndarray)
        assert isinstance(targets, np.ndarray)
        assert (samples.shape[0] == targets.shape[0])
        assert (sample_weight is None) or \
               (isinstance(sample_weight, np.ndarray) and sample_weight.shape[0] == targets.shape[0])

        num_samp, num_feat = samples.shape
        num_pos = np.sum(targets != 0)
        num_neg = np.sum(targets == 0)
        nWeak = self.opts.nWeak[self._stage]
        sample_weight = (np.ones(num_samp, dtype=float) / num_samp) if (sample_weight is None) else sample_weight

        msg = 'Training AdaBoost: nWeak={} nFtrs={} pos={} neg={}'
        if self.opts.verbose:
            print(msg.format(nWeak, num_feat, num_pos, num_neg))

        # limit weight range to [lo_bound, hi_bound]
        lo_bound = 0.0001
        hi_bound = 1.0 - lo_bound
        sample_weight[sample_weight < lo_bound] = lo_bound
        sample_weight[sample_weight > hi_bound] = hi_bound

        # self.detector = AdaBoostClassifier(base_estimator=self._base_estimator, n_estimators=nWeak,
                                                       # algorithm=('SAMME' if self.opts.discrete else 'SAMME.R'))
        # self.detector = GradientBoostingClassifier(learning_rate=0.1, n_estimators=nWeak,
                                                   # max_depth=self.opts.pTree.max_depth, subsample=self.opts.pTree.frac_samp,
                                                   # max_features=self.opts.pTree.frac_feat, verbose=True,
                                                   # min_samples_split=20, min_samples_leaf=10)
        self.detector = RandomForestClassifier(n_estimators=nWeak, criterion='entropy', max_features=self.opts.pTree.frac_feat,
                                               min_samples_split=20, min_samples_leaf=10, verbose=0)

        start = time.clock()
        self.detector.fit(samples, targets, sample_weight)
        end = time.clock()
        print('Done training stage{} in {}s.'.format(self._stage, end - start))

        if save:
            save_name = save_path + 'bf_detector_stage{}.pkl'.format(self._stage)
            joblib.dump(self.detector, save_name)
            print('model saved as {}\n'.format(save_name))

        self._stage += 1
        return

    # train boosted tree for multiple stages
    def train(self, samples, targets, sample_weight=None, save=True, save_path='./'):
        self._stage = 0
        print(self.opts)

        # iterate bootstraping and training
        for i in range(len(self.opts.nWeak)):
            self.train_one_stage(samples, targets, sample_weight, save, save_path)
            accuracy = self.detector.score(samples, targets)
            print('test accuracy = {}\n'.format(accuracy))

        # test each stage model except the last stage
        # if self._stage < len(self.opts.nWeak) - 1:


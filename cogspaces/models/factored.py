import tempfile
import warnings
from math import ceil, floor
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from torch import nn
from torch.autograd import Variable
from torch.nn import Linear
from torch.nn.functional import nll_loss
from torch.optim import Adam
from torch.utils.data import DataLoader

from cogspaces.data import NiftiTargetDataset, infinite_iter


class MultiTaskModule(nn.Module):
    def __init__(self, in_features,
                 shared_embedding_size,
                 target_sizes,
                 input_dropout=0.,
                 dropout=0.,
                 skip_connection=False):
        super().__init__()

        self.in_features = in_features
        self.shared_embedding_size = shared_embedding_size

        self.dropout = nn.Dropout(p=dropout)
        self.input_dropout = nn.Dropout(p=input_dropout)

        self.skip_connection = skip_connection

        if shared_embedding_size > 0:
            self.shared_embedder = Linear(in_features,
                                          shared_embedding_size, bias=False)

        self.classifiers = {}

        for study, size in target_sizes.items():
            self.classifiers[study] = Linear(shared_embedding_size
                                             + in_features * skip_connection,
                                             size)
            self.add_module('classifier_%s' % study, self.classifiers[study])
        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            if param.ndimension() == 2:
                nn.init.xavier_uniform(param)
            elif param.ndimension() == 1:
                param.data.fill_(0.)

    def forward(self, input):
        preds = {}
        for study, sub_input in input.items():
            sub_input = self.input_dropout(sub_input)

            if self.shared_embedding_size > 0:
                embedding = self.dropout(self.shared_embedder(sub_input))
            else:
                embedding = sub_input
            embedding = torch.cat(embedding, dim=1)
            pred = self.classifiers[study](embedding)
            pred = F.softmax(pred, dim=1)
            preds[study] = pred
        return preds

    def coefs(self):
        coefs = {}
        for study in self.classifiers:
            coef = self.classifiers[study][0].weight.data
            if self.shared_embedding_size > 0:
                shared_embed = self.shared_embedder.weight.data
                coef = torch.matmul(coef, shared_embed)
            coefs[study] = coef.transpose(0, 1)
        return coefs

    def intercepts(self):
        return {study: classifier[0].bias.data for study, classifier
                in self.classifiers.items()}


class MultiTaskLoss(nn.Module):
    def __init__(self, study_weights: Dict[str, float])\
            -> None:
        super().__init__()
        self.study_weights = study_weights

    def forward(self, inputs: Dict[str, torch.FloatTensor],
                targets: Dict[str, torch.LongTensor]) -> torch.FloatTensor:
        loss = 0
        for study in inputs:
            study_pred, pred, penalty = inputs[study]
            study_target, target = targets[study][:, 0], targets[study][:, 1]

            loss += nll_loss(pred, target,
                             size_average=True) * self.study_weights[study]
        return loss


def next_batches(data_loaders, cuda, device):
    inputs = {}
    targets = {}
    batch_size = 0
    for study, loader in data_loaders.items():
        input, target = next(loader)
        batch_size += input.shape[0]
        target = target[:, [0, 2]]
        if cuda:
            input = input.cuda(device=device)
            target = target.cuda(device=device)
        input = Variable(input)
        target = Variable(target)
        inputs[study] = input
        targets[study] = target
    return inputs, targets, batch_size


class FactoredClassifier(BaseEstimator):
    def __init__(self,
                 shared_embedding_size=30,
                 batch_size=128,
                 lr=0.001,
                 dropout=0.5, input_dropout=0.25,
                 max_iter=10000, verbose=0,
                 device=-1,
                 seed=None):
        self.shared_embedding_size = shared_embedding_size

        self.input_dropout = input_dropout
        self.dropout = dropout
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose
        self.device = device
        self.lr = lr

        self.seed = seed

    def fit(self, X, y, study_weights=None, callback=None):

        cuda, device = self._check_cuda()
        in_features = next(iter(X.values())).shape[1]
        n_samples = sum(len(this_X) for this_X in X.values())

        if study_weights is None:
            study_weights = {study: 1. for study in X}

        data_loaders = {}
        target_sizes = {}

        torch.manual_seed(self.seed)

        for study in X:
            target_sizes[study] = int(y[study]['contrast'].max()) + 1
            data_loaders[study] = DataLoader(
                NiftiTargetDataset(X[study], y[study]),
                shuffle=True,
                batch_size=self.batch_size, pin_memory=cuda)

        data_loaders = {study: infinite_iter(loader) for study, loader in
                        data_loaders.items()}

        self.module_ = MultiTaskModule(
            in_features=in_features,
            shared_embedding_size=self.shared_embedding_size,
            input_dropout=self.input_dropout,
            dropout=self.dropout,
            target_sizes=target_sizes)
        self.loss_ = MultiTaskLoss(study_weights=study_weights)

        self.optimizer_ = Adam(self.module_.parameters(), lr=self.lr, )

        self.n_iter_ = 0
        # Logging logic
        old_epoch = -1
        seen_samples = 0
        epoch_loss = 0
        epoch_iter = 0
        report_every = ceil(self.max_iter / self.verbose)

        self.module_.train()
        while self.n_iter_ < self.max_iter:
            self.optimizer_.zero_grad()
            inputs, targets, batch_size = next_batches(data_loaders,
                                                       cuda=cuda,
                                                       device=device)
            self.module_.train()
            preds = self.module_(inputs)
            this_loss = self.loss_(preds, targets)
            this_loss.backward()
            self.optimizer_.step()

            seen_samples += batch_size
            epoch_loss += this_loss
            epoch_iter += 1
            self.n_iter_ = seen_samples / n_samples
            epoch = floor(self.n_iter_)
            if report_every is not None and epoch > old_epoch \
                    and epoch % report_every == 0:
                epoch_loss /= epoch_iter
                print('Epoch %.2f, train loss: % .4f'
                      % (epoch, epoch_loss))
                epoch_loss = 0
                epoch_iter = 0
                if callback is not None:
                    callback(self.n_iter_)
            old_epoch = epoch

    def _check_cuda(self):
        if self.device > -1 and not torch.cuda.is_available():
            warnings.warn('Cuda is not available on this system: computation'
                          'will be made on CPU.')
            device = -1
            cuda = False
        else:
            device = self.device
            cuda = device > -1
        return cuda, device

    def predict_proba(self, X):
        cuda, device = self._check_cuda()

        data_loaders = {}
        for study, this_X in X.items():
            data_loaders[study] = DataLoader(NiftiTargetDataset(this_X),
                                             batch_size=len(this_X),
                                             shuffle=False,
                                             pin_memory=cuda)
        preds = {}
        self.module_.eval()
        for study, loader in data_loaders.items():
            pred = []
            for (input, _) in loader:
                if cuda:
                    input = input.cuda(device=device)
                input = Variable(input, volatile=True)
                input = {study: input}
                this_pred = self.module_(input)[study]
                pred.append(this_pred)
            preds[study] = torch.cat(pred)
        preds = {study: pred.data.cpu().numpy()
                 for study, pred in preds.items()}
        return preds

    def predict(self, X):
        preds = self.predict_proba(X)
        preds = {study: np.argmax(pred, axis=1)
                 for study, pred in preds.items()}
        dfs = {}
        for study in preds:
            pred = preds[study]
            dfs[study] = pd.DataFrame(
                dict(contrast=pred, subject=0))
        return dfs

    @property
    def coef_(self):
        coefs = self.module_.coefs()
        return {study: coef.cpu().numpy() for study, coef in coefs.items()}

    @property
    def intercept_(self):
        intercepts = self.module_.intercepts()
        return {study: intercept.cpu().numpy()
                for study, intercept in intercepts.items()}

    @property
    def coef_cat_(self):
        return np.concatenate(list(self.coef_.values()), axis=1)

    @property
    def intercept_cat_(self):
        return np.concatenate(self.intercept_)

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ['module_', 'optimizer_', 'scheduler_']:
            if key in state:
                val = state.pop(key)
                with tempfile.SpooledTemporaryFile() as f:
                    torch.save(val, f)
                    f.seek(0)
                    state[key] = f.read()

        return state

    def __setstate__(self, state):
        disable_cuda = False
        for key in ['module_', 'optimizer_']:
            if key not in state:
                continue
            dump = state.pop(key)
            with tempfile.SpooledTemporaryFile() as f:
                f.write(dump)
                f.seek(0)
                if state['device'] > - 1 and not torch.cuda.is_available():
                    val = torch.load(
                        f, map_location=lambda storage, loc: storage)
                    disable_cuda = True
                else:
                    val = torch.load(f)
            state[key] = val
        if disable_cuda:
            warnings.warn(
                "Model configured to use CUDA but no CUDA devices "
                "available. Loading on CPU instead.")
            state['device'] = -1

        self.__dict__.update(state)

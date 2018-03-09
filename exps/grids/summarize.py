# Baseline logistic
import json
import os
import re
from os.path import expanduser, join

import numpy as np
import pandas as pd
from joblib import load

from cogspaces.introspect.maps import coefs_from_model

import matplotlib.pyplot as plt
import seaborn as sns


def summarize_baseline():
    output_dir = expanduser('~/output/cogspaces/baseline')

    regex = re.compile(r'[0-9]+$')
    res = []
    for this_dir in filter(regex.match, os.listdir(output_dir)):
        this_exp_dir = join(output_dir, this_dir)
        this_dir = int(this_dir)
        try:
            config = json.load(
                open(join(this_exp_dir, 'config.json'), 'r'))
            run = json.load(open(join(this_exp_dir, 'run.json'), 'r'))
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            print('Skipping exp %i' % this_dir)
            continue
        study = config['data']['studies']
        if run['result'] is None:
            continue
        else:
            test_score = run['result'][study]
        res.append(dict(study=study, test_score=test_score,
                        run=this_dir))
    res = pd.DataFrame(res)

    max_res = res.groupby(by='study').aggregate('idxmax')['test_score']
    max_res = res.iloc[max_res.values.tolist()]
    pd.to_pickle(max_res, join(expanduser('~/output/cogspaces/'
                                          'max_baseline.pkl')))


def summarize_factored():
    output_dir = [expanduser('~/output/cogspaces/factored'), ]

    regex = re.compile(r'[0-9]+$')
    res = []
    for this_output_dir in output_dir:
        for this_dir in filter(regex.match, os.listdir(this_output_dir)):
            this_exp_dir = join(this_output_dir, this_dir)
            this_dir = int(this_dir)
            try:
                config = json.load(
                    open(join(this_exp_dir, 'config.json'), 'r'))
                run = json.load(open(join(this_exp_dir, 'run.json'), 'r'))
                info = json.load(
                    open(join(this_exp_dir, 'info.json'), 'r'))
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                print('Skipping exp %i' % this_dir)
                continue
            estimator = config['model']['estimator']
            studies = config['data']['studies']
            test_scores = run['result']
            if test_scores is None:
                test_scores = info['test_scores'][-1]
            this_res = dict(estimator=estimator,
                            run=this_dir)
            this_res['study_weight'] = config['model']['study_weight']
            this_res['shared_embedding_size'] = config['factored'][
                'shared_embedding_size']
            this_res['dropout'] = config['factored']['dropout']
            this_res['input_dropout'] = config['factored'][
                'input_dropout']
            this_res['lr'] = config['factored']['lr']
            if studies == 'all' and test_scores is not None:
                mean_test = np.mean(np.array(
                    list(test_scores.values())))
                this_res['mean_test'] = mean_test
                this_res = dict(**this_res, **test_scores)
                res.append(this_res)
    res = pd.DataFrame(res)
    res.set_index(['private_embedding_size',
                   'dropout', 'input_dropout', 'lr', 'estimator', 'run',
                   'study_weight'], inplace=True)
    res = res.sort_index()
    pd.to_pickle(res, join(expanduser('~/output/cogspaces/factored.pkl')))
    max = res.apply('max')
    pd.to_pickle(max, join(expanduser('~/output/cogspaces/max_factored.pkl')))


def plot():
    output_dir = expanduser('~/output/cogspaces/')
    baseline = pd.read_pickle(join(output_dir, 'max_baseline.pkl'))
    baseline = baseline.drop('run', axis=1)
    baseline = baseline.set_index('study')
    factored = pd.read_pickle(join(output_dir, 'max_factored.pkl'))
    factored.name = 'factored'
    res = baseline.join(factored)
    res = res.rename({'test_score': 'baseline'},
                               axis='columns')
    res['diff'] = res['factored'] - res['baseline']
    res = res.sort_values('diff', ascending=False)
    print(res)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    n_study = res.shape[0]
    ind = np.arange(n_study) * 2
    width = .8
    rects1 = ax.bar(ind, res['baseline'], width)
    rects2 = ax.bar(ind + width, res['factored'], width)
    ax.set_ylabel('Test accuracy')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(res.index.values, rotation=60, ha='right',
                       va='top')
    ax.set_ylim([0.16, 0.92])

    ax.legend((rects1[0], rects2[0]), ('Baseline', 'Factored'))
    plt.subplots_adjust(top=0.98, bottom=0.32, left=0.07, right=0.98)
    sns.despine(fig)
    plt.savefig(join(output_dir, 'comparison.pdf'))
    plt.show()


if __name__ == '__main__':
    # summarize_mtl
    # summarize_baseline()
    summarize_factored()
    # plot()

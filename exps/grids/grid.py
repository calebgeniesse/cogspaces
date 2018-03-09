from os.path import join

import numpy as np
import sys
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid

from cogspaces.datasets.utils import get_data_dir, get_output_dir
from cogspaces.utils.sacred import get_id, OurFileStorageObserver
from exps.train import exp


@exp.config
def base():
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512_lstsq'),
        studies='all'
    )


def trace():
    model = dict(
        normalize=True,
        estimator='trace',
        study_weight='sqrt',
        max_iter=300,
    )
    trace = dict(
        trace_penalty=5e-2,
    )


def factored_dropout():
    model = dict(
        normalize=True,
        estimator='factored',
        max_iter=300,
    )
    factored = dict(
        optimizer='sgd',
        shared_embedding_size=100,
        private_embedding_size=0,
        shared_embedding='hard+adversarial',
        skip_connection=False,
        batch_size=128,
        dropout=0.75,
        lr=1e-2,
        input_dropout=0.5,
    )


def factored_l2():
    model = dict(
        normalize=True,
        estimator='factored',
        max_iter=300,
    )

    factored = dict(
        optimizer='sgd',
        shared_embedding_size=100,
        private_embedding_size=0,
        shared_embedding='hard+adversarial',
        skip_connection=False,
        batch_size=128,
        dropout=0.75,
        lr=1e-2,
        input_dropout=0.5,
    )


def factored():
    system = dict(
        device=-1,
        seed=0,
        verbose=50,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512'),
        studies='all'
    )

    model = dict(
        normalize=True,
        estimator='factored',
        study_weight='study',
        max_iter=500,
    )
    factored = dict(
        shared_embedding_size=100,
        batch_size=32,
        dropout=0.75,
        lr=1e-3,
        input_dropout=0.25,
    )


def run_exp(output_dir, config_updates, _id):
    """Boiler plate function that has to be put in every multiple
        experiment script, as exp does not pickle."""
    exp.run_command('print_config', config_updates=config_updates, )
    run = exp._create_run(config_updates=config_updates, )
    run._id = _id
    observer = OurFileStorageObserver.create(basedir=output_dir)
    run.observers.append(observer)
    run()


if __name__ == '__main__':
    output_dir = join(get_output_dir(), 'factored')
    exp.config(factored)
    config_updates = []
    config_updates += list(
        ParameterGrid({'data.studies': [['archi'],
                                            ['archi', 'hcp'],
                                            ['brainomics'],
                                            ['brainomics', 'hcp'],
                                            ['camcan'],
                                            ['camcan', 'hcp'],
                                            ['la5c'],
                                            ['la5c', 'hcp']],
                       }))
    _id = get_id(output_dir)
    Parallel(n_jobs=32, verbose=100)(delayed(run_exp)(output_dir,
                                                      config_update,
                                                      _id=_id + i)
                                     for i, config_update
                                     in enumerate(config_updates))

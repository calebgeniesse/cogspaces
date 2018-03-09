# Cogspaces
Leverage multiple fMRI studies to predict cognitive states from brain maps.

Implements the following paper ([PDF Link](http://papers.nips.cc/paper/7170-learning-neural-representations-of-human-cognition-across-many-fmri-studies.pdf))

> Learning Neural Representations of Human Cognition across Many fMRI Studies
> 
> A Mensch, J Mairal, D Bzdok, B Thirion, G Varoquaux - Advances in Neural Information Processing Systems, 2017

## Install
`python setup.py install`

## Run

So far the reduced task fMRI datasets are not released yet so some adaptations
are needed to use your own data.

`python exps/train.py`

Data reduction (1st layer in the paper) can be done using the
[modl package](http://github.com/arthurmensch/modl).

Grids can be run using

`python exps/grids/grid.py`
 
 Again, adaptation will have to be made for the moment.
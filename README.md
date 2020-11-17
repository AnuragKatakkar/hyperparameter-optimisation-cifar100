# 10605-fall20
Mini Project for CMU 10605 Fall 2020

A quick summary of this implementation
### Utils.py
This file defines the set of HPs and their values to perform a "grid search over"
#### make_configs()
The function "make_configs" returns all hyperparameter configurations as a list 
of namedtuples. Each namedtuple has the following attributes:
1. Name -  a string of the form conf_<lr>_<bs>_<optim>_<momemntum>_<wd>,
2. Model - A densenet121 model,
3. Dataloader - a *training* dataloader with the required batch size: since there
are only 3 batch sizes, we share dataloaders among configurations,
4. Optimizer - the respective optimiser with HP setting

#### train_model()
Performs a set number of epochs for training

### model.py
This consists the DenseNet121 implementation

### main.py
This is where we 
1. Get the list of all configs (81),
2. Define test dataloader and criterion,
3. Implement the respective HP Optimisation technique and train
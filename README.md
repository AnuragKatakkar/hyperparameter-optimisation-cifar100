# Hyperband for Image Classification on CIFAR-100
This repository is meant to serve as an example of how Hyperband [paper](https://arxiv.org/abs/1603.06560) can be used to find the best set of hyperparameters for achieving >90% accuracy on the CIFAR-100 benchmark using a DenseNet-121 model.

The actual implementation of the Hyperband Algorithm is owed to Kevin Jamieson, from his blog [here](https://homes.cs.washington.edu/~jamieson/hyperband.html), and the PyTorch implementation of DenseNet-121 to this [repository](https://github.com/kuangliu/pytorch-cifar/blob/master/models/densenet.py).

## An Overview of This Implementation
As explained in the blog above, we need a few things to get a working implementation of Hyperband:
1. An ML/DL model that can be trained in an iterative manner,
2. A method that samples random hyperparameter configurations for said model,
3. A method that can train the model for "n" epochs and return the validation loss (or any other accuracy metric that you want to use for successive halving)

### What this repo provides:
1. In *model.py*, we write an implementation of the DenseNet-121 model,
2. In *utils.py*, we define a **get_config()** function that samples a set of hyperparameters, instantiates a model, and returns an instance of the HP_Config class that encapsulates all of this conveniently,
3. Also in *utils.py* we define the training and validation routine,
4. *main.py* : This is where the magic happens. Here, we initialize other boilplate code like PyTorch Dataloaders, and call the Hyperband routine with a choice of number of iterations of successive halving to run, and the maximum number of epochs to run any single model for. 

### Choice of Hyperparameters
In this implementation, we focus on the 5 main Hyperparameters for any Deep Learning Experiment:
1. Learning Rate, (sampled from the range [1e-3, 1e-1])
2. Batch Size, (sampled from (32, 64, 128))
3. Optimizer (we consider Adam, SGD, and RMSprop),
4. Momentum, (sampled from the range [0.8, 0.9]) and,
5. Weight Decay (sampled from the range [5e-6, 6e-6])

Users are free to define their own sampling techniques for these Hyperparameters. Note that the sampling technique greatly affects performance!

## Use this repo for your own AutoML Experiments!
If you are interested in training models to achieve competitive performance on CIFAR-100 without having to worry about choosing the best set of Hyperparameters by hand, this repo is for you!

Here are a few steps you can follow to get started:
1. Add other models to *model.py*. Update *utils.py* accordingly to import this model and initialize it in **get_config()**. This can be any model that can be used for image classification and not specific to PyTorch (although using other models will require redefining the dataloaders as well),
2. Set the Hyperband hyperparameters (s, max_iter) in main.py,
3. Optional : Experimient with other sampling methods in **get_config()** though we recommend using the given implementation for best results.

## Coming Soon
1. An implementation of **run_hyperband()** that abstracts away the actual hyperband implementation,
2. Extensive code documentation.

## Dependencies
The following dependencies need to be installed 
```bash
pip install torch numpy tqdm torchvision matoplotlib
```

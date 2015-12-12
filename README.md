## Overview

This project is completed in partial requirement for ECE469 Artificial Intelligence at The Cooper Union. It is a C++ implementation of a 3 layer neural network.

<!-- MarkdownTOC -->

- [Instructions](#instructions)
	- [Build](#build)
	- [Usage](#usage)
		- [File Format](#file-format)
			- [Neural Net](#neural-net)
			- [Dataset](#dataset)
		- [Training](#training)
		- [Testing](#testing)
		- [Randomizer](#randomizer)

<!-- /MarkdownTOC -->

## Instructions

### Build

To build this project, you can run the following commands
```
git clone https://github.com/linkelvin11/nn.git nn;
cd nn;
make;
./nn.exe;

```

### Usage

#### File Format

##### Neural Net

This program loads a neural net from file.

The first line of the file must contain 3 integers in the following order:
Number of input nodes (N_i)
Number of hidden nodes (N_h)
Number of output nodes (N_o)

The next N_h lines will contain the weights from each input node to a given hidden node. For example, the first line will contain the weights from each input node to the first hidden node. Each weight is represented with a fixed point number with 3 digits after the decimal. The first weight in each line will represent the bias for that node. As a result, each of these N_h lines will have N_i + 1 weights.

The next N_o lines will contain the weights from each hidden node to a given output node. The formatting is the same as the previous N_h lines; each line will contain N_h + 1 weights, with the first weight representing the bias of that node.

##### Dataset

The dataset must follow this format for both training and testing sets:

The first line of the file must contain 3 integers in the following order:
Number of samples (N_s)
Number of features (N_i)
Number of labels (N_o)

As you can see, the number of features and labels must match the number of inputs and outputs used for the neural net.

Each line of the dataset file other than the first represents a sample. The first N_i numbers on each line represent features. While not absolutely necessary, it is recommended that these features are normalized from 0 to 1. The next N_o numbers on that same line represent the labels for that sample. Each label must be either 0 or 1.

#### Training

To train the neural net, 2 files are required: a representation of the neural net and a training set. If a neural net file is not available, you can use the [randomizer](#randomizer).

Follow the prompts from the program to train your network.

#### Testing

To use the trained neural net, 2 files are required: a representation of the neural net and a test set.

Follow the prompts from the program to train your network.

#### Randomizer

The randomizer allows you to initialize a neural network with random weights. When the program prompts you `would you like to train or test? (train/test)` do not enter either choice. Instead, enter `gen`. The program will then ask for parameters for your neural net. Enter them as instructed.

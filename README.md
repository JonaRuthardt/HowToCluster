# Using Compute Clusters for Deep Learning

👋 Welcome! This repository is a complementary resource to [this tutorial](https://jonaruthardt.github.io/docs/cluster_tutorial/) on setting up and using clusters for deep learning workflows. In this repository, you’ll find some additional resources and examples to help you:

1. Setup your environment.
2. Schedule jobs efficiently using a job scheduler (e.g., SLURM).
3. Run, debug and monitor your experiments.

## Contents
- `install_environment.job`: A job script to setup a conda env and install basic packages
- `mnist_classifier.py`: A simple digit classification model
- `run_experiment.job`: Implement this file to start training the MNIST model

## First Steps
Make sure you already followed the steps on cluster access and authentication outlined in [our guide](https://jonaruthardt.github.io/docs/cluster_tutorial/) and can now sucessfully connect to the cluster.

### Setting up the Code
Clone/fork the repository so that you can execute files on the cluster and can configure and edit the scripts remotely (or locally).

### Setting up the Environment

After you connected to the cluster via ssh, it is time to setup the environment you'll be working in. 

**Your task:** Submit the `install_environment.job` job that creates the conda environment and installs the required libraries. 

After submitting, you can check whether your job is still in the queue, currently running, or already finished using the `squeue` command.
Verify that the job ran correctly by looking at the job's output file. Now, you're ready to use this conda environment for your subsequent experiments. 

## Running Experiments
Next, we want to use the cluster to run some experiments. As an example, you should train a simple neural network on the MNIST dataset for digit classification. You will find the full implementation in the `mnist_classifier.py` file.

**Your task:** Complete/fill in the `run_experiment.job` file such that one GPU is requested and the MNIST model is trained and evaluated. 
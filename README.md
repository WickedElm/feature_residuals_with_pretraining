# Autoencoder Feature Residuals for Network Intrusion Detection:  Unsupervised Pre-training for Improved Performance 

This repo contains the code needed to support the paper titled "Autoencoder Feature Residuals for Network Intrusion Detection:  Unsupervised Pre-training for Improved Performance". Note that this code contains dependencies on external services such as Weights and Biases and other python libraries.

## Extended book chapter code for recurrent neural networks
The code for the extended book chapter for recurrent neural networks can be found here:  [https://github.com/WickedElm/feature_residuals_with_pretraining_rnn](https://github.com/WickedElm/feature_residuals_with_pretraining_rnn)

# Executing Code
Assuming all of the dependencies are in place such as an account on Weights and Biases one can execute experiments by each dataset or all at once.

## Running an experiment for all datasets

To do this simply clone the repo, change to its directory, and execute:

```
./run_all_tests
```

This will download the data needed and execute what we considered a single experiment in the paper.

## Running an experiment for a specific dataset
To do this, clone the repo, change to its directory, and execute:

```
./download_data
./run_<dataset to run>
```

where you replace with one of the dataset run scripts in the repo such as run_nf_unsw_nb15_v2

## Finding results data
After performing an experiment the resulting data can be found in the ./outputs/ directory.

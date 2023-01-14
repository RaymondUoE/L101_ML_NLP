# L101 Machine Learning for Language Processing

This is a proejct for L101 Machine Learning for Language Processing module at the University of Cambridge, MPhil ACS 2022/23 academic year. 


## Data required

It is requied to have SNLI, MNLI, and ChaosNLI data, stored at`./data/snli`, `./data/mnli`and`./data/chaosNLI`respectively.

## Files

 - `dataset.py`-- Custom dataset object to handle soft labels
 - `fine_tune.py`-- Pipeline for fine-tuning sentence transformers
 - `model.py`-- Model file for classifier module
 - `My....py`-- Classes inherenting default sentence transformer classes to support soft labels
 - `preprocessing.py`-- Preprocessing
 - `train.py`Training pipeline for the classifier module
 - `utils.py`-- Utilities

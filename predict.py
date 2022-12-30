import torch
from torch import nn
import pandas as pd
from utils import *
from dataset import CustomEmbeddingDataset
import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# from sentence_transformers import InputExample, models, losses, evaluation #SentenceTransformer
# from MySentenceTransformer import MySentenceTransformer
# from torch.utils.data import DataLoader
# from MySoftmax import MySoftmaxLoss
# from MyInputExample import MyInputExample
# from MyLabelAccuracyEvaluator import MyLabelAccuracyEvaluator
import pickle

def main():
    with open(f'model/tuning_vanilla_fine/prediction/predicted.pkl', "rb") as f:
            lists = pickle.load(f)
            f.close() 
    print(torch.cat(lists, dim=0).cpu().detach().numpy())
    
    
if __name__ == "__main__":
    main()
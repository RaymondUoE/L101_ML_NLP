import torch
from torch import nn
import pandas as pd
from utils import *
from dataset import CustomEmbeddingDataset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import InputExample, models, losses, evaluation #SentenceTransformer
from MySentenceTransformer import MySentenceTransformer
from torch.utils.data import DataLoader
from MySoftmax import MySoftmaxLoss
from MyInputExample import MyInputExample
from MyLabelAccuracyEvaluator import MyLabelAccuracyEvaluator

def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    
    model = MySentenceTransformer('model/tune').to(device)
    print(model.children)



if __name__ == "__main__":
    main()
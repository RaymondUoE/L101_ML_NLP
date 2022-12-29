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
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    # model_id = 'sentence-transformers/distilbert-base-nli-mean-tokens'
    # model_id = 'sentence-transformers/roberta-base-nli-mean-tokens'
    model = MySentenceTransformer(model_id).to(device)
    
    # load coarse data
    train_data = pd.read_csv('data/preprocessed/nli_coarse_train.csv')
    train_data['label_count'] = train_data.apply(lambda x: [int(y) for y in string_of_list_to_list(x['label_count'])], axis=1)
    train_data['label_dist'] = train_data.apply(lambda x: [float(y) for y in string_of_list_to_list(x['label_dist'])], axis=1)
    val_data = pd.read_csv('data/preprocessed/nli_coarse_val.csv')
    val_data['label_count'] = val_data.apply(lambda x: [int(y) for y in string_of_list_to_list(x['label_count'])], axis=1)
    val_data['label_dist'] = val_data.apply(lambda x: [float(y) for y in string_of_list_to_list(x['label_dist'])], axis=1)
    
    num_of_labels = len(train_data['label_dist'][0])
    # add converted score
    # train_data = add_annotation_info(train_data)
    # train_data = train_data[train_data['num_of_annotation']==5].copy().reset_index(drop=True)
    # val_data = add_annotation_info(val_data)
    # val_data = val_data[val_data['num_of_annotation']==5].copy().reset_index(drop=True)
    
    # build examples
    train_examples = build_examples(train_data, seed=4)
    val_examples = build_examples(val_data, seed=5)
    
    train_dataloader = DataLoader(train_examples, batch_size=32)
    val_dataloader = DataLoader(val_examples, batch_size=32)
    
    # train_loss = losses.CosineSimilarityLoss(model)
    train_loss = MySoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=num_of_labels)
    
    evaluator = MyLabelAccuracyEvaluator(dataloader=val_dataloader, softmax_model=train_loss)
    
    model.fit(train_objectives=[(train_dataloader, train_loss)], 
              evaluator=evaluator,
              epochs=1, 
              warmup_steps=100,
              evaluation_steps=500,
              output_path='model/tune',
              save_best_model=True)

def add_annotation_info(df):
    df['num_of_annotation'] = df.apply(lambda x: np.sum(x['label_count']), axis=1)
    df['converted_score'] = df.apply(lambda x: (x['label_count'][0]*1+x['label_count'][1]*0+x['label_count'][2]*(-1)) / np.sum(x['label_count']), axis=1)
    return df
    
def build_examples(df, seed=0):    
    # np.random.seed(seed)
    # noises = np.random.normal(0, 0.05, len(df))
    # examples = []
    # for i, row in df.iterrows():
    #     examples.append(InputExample(texts=[row['premise'], row['hypothesis']], label=float(row['converted_score']+noises[i])))
    # return examples
    examples = []
    for i, row in df.iterrows():
        examples.append(MyInputExample(texts=[row['premise'], row['hypothesis']], label=torch.tensor(row['label_dist'])))
    return examples
    
    
if __name__ == "__main__":
    main()
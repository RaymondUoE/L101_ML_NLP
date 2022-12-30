import torch
from torch import nn
import pandas as pd
from utils import *
# from dataset import CustomEmbeddingDataset
# import torch.nn.functional as F
import numpy as np
# import matplotlib.pyplot as plt
# from sentence_transformers import InputExample, models, losses, evaluation #SentenceTransformer
from MySentenceTransformer import MySentenceTransformer
from torch.utils.data import DataLoader
from MySoftmax import MySoftmaxLoss
from MyInputExample import MyInputExample
from MyLabelAccuracyEvaluator import MyLabelAccuracyEvaluator
import os

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)
model_id = "sentence-transformers/all-MiniLM-L6-v2"
# model_id = 'sentence-transformers/roberta-base-nli-mean-tokens'
# model_id = "sentence-transformers/all-mpnet-base-v2"
HIDDEN_DIM = 512
BATCH_SIZE = 64


def main():
    
    model_out_path = 'model/mini_R1'
    train_data_path = 'data/preprocessed/nli_coarse_train.csv'
    val_data_path = 'data/preprocessed/nli_coarse_val.csv'
    fine_tune(model_id, device, model_out_path, train_data_path, val_data_path, epoch=1, round=1)
    
    model_out_path = 'model/mini_R2'
    train_data_path = 'data/preprocessed/chaos_train.csv'
    val_data_path = 'data/preprocessed/chaos_val.csv'
    test_data_path = 'data/preprocessed/chaos_test.csv'
    
    fine_tune(model_id, device, model_out_path, train_data_path, val_data_path, test_data_path=test_data_path, epoch=1, predict=True, round=2)

    

def fine_tune(model_id, device, model_out_path, train_data_path, val_data_path, epoch, round, test_data_path=None, predict=False):
    model = MySentenceTransformer(model_id).to(device)
    train_data = pd.read_csv(train_data_path)
    train_data['label_count'] = train_data.apply(lambda x: [int(y) for y in string_of_list_to_list(x['label_count'])], axis=1)
    train_data['label_dist'] = train_data.apply(lambda x: [float(y) for y in string_of_list_to_list(x['label_dist'])], axis=1)
    val_data = pd.read_csv(val_data_path)
    val_data['label_count'] = val_data.apply(lambda x: [int(y) for y in string_of_list_to_list(x['label_count'])], axis=1)
    val_data['label_dist'] = val_data.apply(lambda x: [float(y) for y in string_of_list_to_list(x['label_dist'])], axis=1)
    if test_data_path:
        test_data = pd.read_csv('data/preprocessed/chaos_test.csv')
        test_data['label_count'] = test_data.apply(lambda x: [int(y) for y in string_of_list_to_list(x['label_count'])], axis=1)
        test_data['label_dist'] = test_data.apply(lambda x: [float(y) for y in string_of_list_to_list(x['label_dist'])], axis=1)
        test_examples = build_examples(test_data)
        test_dataloader = DataLoader(test_examples, batch_size=BATCH_SIZE)
    
    num_of_labels = len(train_data['label_dist'][0])
    
    # build examples
    train_examples = build_examples(train_data)
    val_examples = build_examples(val_data)
    
    train_dataloader = DataLoader(train_examples, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_examples, batch_size=BATCH_SIZE)
    
    train_loss = MySoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=num_of_labels, round=round, hidden=HIDDEN_DIM)
    
    evaluator = MyLabelAccuracyEvaluator(dataloader=val_dataloader, softmax_model=train_loss)
    
    model.fit(train_objectives=[(train_dataloader, train_loss)], 
              evaluator=evaluator,
              epochs=epoch, 
              warmup_steps=100,
              output_path=model_out_path,
              save_best_model=True)
    
    if predict:
        evaluator.eval_model(train_loss, val_dataloader, output_path=os.path.join(model_out_path, 'prediction'))
        
    
def add_annotation_info(df):
    df['num_of_annotation'] = df.apply(lambda x: np.sum(x['label_count']), axis=1)
    df['converted_score'] = df.apply(lambda x: (x['label_count'][0]*1+x['label_count'][1]*0+x['label_count'][2]*(-1)) / np.sum(x['label_count']), axis=1)
    return df
    
def build_examples(df):
    examples = []
    for i, row in df.iterrows():
        examples.append(MyInputExample(texts=[row['premise'], row['hypothesis']], label=torch.tensor(row['label_dist'])))
    return examples
    
    
if __name__ == "__main__":
    main()
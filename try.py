from sentence_transformers import InputExample, models, losses #SentenceTransformer
from MySentenceTransformer import SentenceTransformer
import torch
from torch import nn
import os
from torch.utils.data import DataLoader
from model import Classifier, NLIClassify
import pytorch_lightning as pl
from dataset import CustomEmbeddingDataset
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import pickle
from MySoftmax import SoftmaxLoss
from MyInputExample import MyInputExample



device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

model_id = "sentence-transformers/all-MiniLM-L6-v2"
sent_t = SentenceTransformer(model_id, device=device)

def main():
    # train_dataset = check_saved_data(mode='train')
    # val_dataset = check_saved_data(mode='val')
    # test_dataset = check_saved_data(mode='test')
    
    train_examples = [MyInputExample(texts=['My first sentence', 'My second sentence'], label=torch.tensor([0.2, 0.8, 0])),
    MyInputExample(texts=['Another pair', 'Unrelated sentence'], label=torch.tensor([0.1, 0.5, 0.4]))]

    #Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    # train_loss = losses.CosineSimilarityLoss(model)
    train_loss = SoftmaxLoss(model=sent_t, sentence_embedding_dimension=sent_t.get_sentence_embedding_dimension(), num_labels=3)

    #Tune the model
    sent_t.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)



def check_saved_data(mode):
    if mode not in ['train', 'val', 'test']:
        raise Exception('Mode error. train val test')
    if os.path.exists(f'./data/processed/dataset_{mode}.pkl'):
        print(f'Data found, mode = {mode}. Loading...')
        with open(f'./data/processed/dataset_{mode}.pkl', 'rb') as f:
            dataset = pickle.load(f)
            f.close()
    else:
        dataset = CustomEmbeddingDataset(sentences_file=f'data/preprocessed/chaos_{mode}.csv', encoder=sent_t, device=device)
        with open(f'./data/processed/dataset_{mode}.pkl', 'wb') as f:
            pickle.dump(dataset, f)
            f.close()
    return dataset


if __name__ == "__main__":
    main()

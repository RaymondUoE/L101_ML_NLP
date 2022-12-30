from sentence_transformers import SentenceTransformer, InputExample, models
import torch
from torch import nn
import os
from torch.utils.data import DataLoader
from model import Classifier, NLIClassify
import pytorch_lightning as pl
from dataset import CustomEmbeddingDataset
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import pickle

CHECKPOINT_PATH = "./model"

MAX_EPOCH = 100
BATCH_SIZE = 16

FEATURE_DIM = 0
CLASS_DIM = 0
torch.manual_seed(22)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# model_id = "sentence-transformers/all-MiniLM-L6-v2"
model_id = "./model/tuning_vanilla_"
sent_t = SentenceTransformer(model_id, device=device)
model_name = 'mini_tuned_on_soft'

def main():
    # train_dataset = CustomEmbeddingDataset(sentences_file='data/preprocessed/chaos_train.csv', encoder=sent_t, device=device)
    # val_dataset = CustomEmbeddingDataset(sentences_file='data/preprocessed/chaos_val.csv', encoder=sent_t, device=device)
    # test_dataset = CustomEmbeddingDataset(sentences_file='data/preprocessed/chaos_test.csv', encoder=sent_t, device=device)
    # train_dataset_coarse = check_saved_data(mode='train', grain='coarse')
    # val_dataset_coarse = check_saved_data(mode='val', grain='coarse')
    train_dataset_fine = check_saved_data(mode='train', grain='fine', model=model_name)
    val_dataset_fine = check_saved_data(mode='val', grain='fine', model=model_name)
    test_dataset_fine = check_saved_data(mode='test', grain='fine', model=model_name)


    global FEATURE_DIM 
    FEATURE_DIM = train_dataset_fine[0]['p'].shape[0]
    global CLASS_DIM 
    CLASS_DIM = train_dataset_fine[0]['label'].shape[0]
    classify_model, result = train_classifier(
                                                    train_dataset=train_dataset_fine,
                                                    val_dataset=val_dataset_fine,
                                                    test_dataset=test_dataset_fine,
                                                    dp_rate=0.1)
    
    
    
    print_results(result)
    

def train_classifier(train_dataset, val_dataset, test_dataset, **model_kwargs):
    model_name = 'Linear'
    pl.seed_everything(67)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    root_dir = os.path.join(CHECKPOINT_PATH, model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_accuracy")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=MAX_EPOCH,
                         enable_progress_bar=True)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    model = NLIClassify(model_name=model_name, 
                         batch_size=BATCH_SIZE, 
                         embed_dim=FEATURE_DIM,
                         h_dim=256,
                         out_dim=CLASS_DIM,
                         **model_kwargs)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    model = NLIClassify.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)


    train_result = trainer.test(model, dataloaders=train_dataloader)[0]
    val_result = trainer.test(model, dataloaders=val_dataloader)[0]
    test_result = trainer.test(model, dataloaders=test_dataloader)[0]
    result = {"train": train_result['test_accuracy'],
              "val": val_result['test_accuracy'],
              "test": test_result['test_accuracy']}
    return model, result

def check_saved_data(mode, grain, model):
    save_path = os.path.join('./data/embedding/', model, f'dataset_{mode}_{grain}.pkl')
    if os.path.exists(save_path):
        print(f'Data found, mode = {mode}, {grain}. Loading...')
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
            f.close()
    else:
        if grain == 'coarse':
            path = f'data/preprocessed/nli_{grain}_{mode}.csv'
        elif grain == 'fine':
            path = f'data/preprocessed/chaos_{mode}.csv'
        else:
            raise Exception('File not found')
        dataset = CustomEmbeddingDataset(sentences_file=path, encoder=sent_t, device=device)
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
            f.close()
    return dataset

def print_results(result_dict):
    if "train" in result_dict:
        print(f"Train accuracy: {(100.0*result_dict['train']):4.2f}%")
    if "val" in result_dict:
        print(f"Val accuracy:   {(100.0*result_dict['val']):4.2f}%")
    print(f"Test accuracy:  {(100.0*result_dict['test']):4.2f}%")




if __name__ == "__main__":
    main()
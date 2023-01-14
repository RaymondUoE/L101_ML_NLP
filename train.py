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
import json

CHECKPOINT_PATH = "./model"

MAX_EPOCH = 100
BATCH_SIZE = 16

FEATURE_DIM = 0
CLASS_DIM = 0
torch.manual_seed(22)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

model_names = [
    'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers/roberta-base-nli-mean-tokens',
    'sentence-transformers/all-mpnet-base-v2',
    'mini_R1_hard',
    'mini_R1_soft',
    'mini_R2_SKIP_soft',
    'mini_R2_hard_soft',
    'mini_R2_soft_soft',
    'mpnet_R1_hard',
    'mpnet_R1_soft',
    'mpnet_R2_SKIP_soft',
    'mpnet_R2_hard_soft',
    'mpnet_R2_soft_soft',
    'robertaNLI_R1_hard',
    'robertaNLI_R1_soft',
    'robertaNLI_R2_SKIP_soft',
    'robertaNLI_R2_hard_soft',
    'robertaNLI_R2_soft_soft',
]

def main(model_name):
    model_name = model_name
    train_dataset = check_saved_data(mode='train', model_name=model_name)
    val_dataset = check_saved_data(mode='val', model_name=model_name)
    test_dataset = check_saved_data(mode='test', model_name=model_name)


    global FEATURE_DIM 
    FEATURE_DIM = train_dataset[0]['p'].shape[0]
    global CLASS_DIM 
    CLASS_DIM = train_dataset[0]['label'].shape[0]

    classify_model, result = train_classifier(
                                                    train_dataset=train_dataset,
                                                    val_dataset=val_dataset,
                                                    test_dataset=test_dataset,
                                                    dp_rate=0.1)
    
    print_results(result)
    
    # inference
    pred_model = classify_model.get_model()
    pred_model.eval().to(device)
    preds = []
    labels = []
    for d in val_dataset:
        # print(d['p'].unsqueeze(1).shape)
        pred = pred_model(d['p'].unsqueeze(0), d['h'].unsqueeze(0))
        preds.append(pred.cpu().detach().numpy())
        labels.append(d['label'].cpu().detach().numpy())
    
    pred_path = os.path.join('predictions', model_name)
    os.makedirs(pred_path, exist_ok=True)
    with open(os.path.join(pred_path, 'preds.pkl'), 'wb') as f:
        pickle.dump(preds, f)
        f.close()
    with open(os.path.join(pred_path, 'labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)
        f.close()
    with open(os.path.join(pred_path, 'results.txt'), 'w') as f:
        json.dump(result, f)
        f.close()
    
    

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

def check_saved_data(mode, model_name):
    save_path = os.path.join('./data/embedding/', model_name)
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f'dataset_chaos_{mode}.pkl')
    if os.path.exists(save_path):
        print(f'Data found, mode = {mode}. Loading...')
        with open(save_path, 'rb') as f:
            dataset = pickle.load(f)
            f.close()
    else:
        data_path = f'data/preprocessed/chaos_{mode}.csv'
        if not model_name.startswith('sentence-transformer'):
            model_path = os.path.join('model', model_name)
        else:
            model_path = model_name
        model = SentenceTransformer(model_path, device=device)
        dataset = CustomEmbeddingDataset(sentences_file=data_path, encoder=model, device=device)
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
    for m in model_names:
        # print(m)
        main(m)
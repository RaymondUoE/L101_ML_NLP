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
BATCH_SIZE = 32

FEATURE_DIM = 0
CLASS_DIM = 0
torch.manual_seed(22)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

model_id = "sentence-transformers/all-MiniLM-L6-v2"
sent_t = SentenceTransformer(model_id, device=device)

# word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# # dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())

# model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

# model = nn.Sequential(
#             *sent_t.children(),
#             nn.Conv2d(1,20,5),
#             nn.ReLU(),
#             nn.Conv2d(20,64,5),
#             nn.ReLU()
#         )
def main():
    # train_dataset = CustomEmbeddingDataset(sentences_file='data/preprocessed/chaos_train.csv', encoder=sent_t, device=device)
    # val_dataset = CustomEmbeddingDataset(sentences_file='data/preprocessed/chaos_val.csv', encoder=sent_t, device=device)
    # test_dataset = CustomEmbeddingDataset(sentences_file='data/preprocessed/chaos_test.csv', encoder=sent_t, device=device)
    train_dataset = check_saved_data(mode='train')
    val_dataset = check_saved_data(mode='val')
    test_dataset = check_saved_data(mode='test')


    global FEATURE_DIM 
    FEATURE_DIM = train_dataset[0]['p'].shape[0]
    global CLASS_DIM 
    CLASS_DIM = train_dataset[0]['label'].shape[0]
    classify_model = train_classifier(
                                                    train_dataset=train_dataset,
                                                    val_dataset=val_dataset,
                                                    test_dataset=test_dataset,
                                                    dp_rate=0.1)
    # print_results(node_gnn_result)

def train_classifier(train_dataset, val_dataset, test_dataset, **model_kwargs):
    model_name = 'Linear'
    pl.seed_everything(67)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    root_dir = os.path.join(CHECKPOINT_PATH, model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=MAX_EPOCH,
                         enable_progress_bar=True)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    model = NLIClassify(model_name=model_name, 
                         batch_size=BATCH_SIZE, 
                         embed_dim=FEATURE_DIM,
                         h_dim=512,
                         out_dim=CLASS_DIM,
                         **model_kwargs)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    model = NLIClassify.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model
    train_result = trainer.test(model, dataloaders=train_dataloader)[0]
    val_result = trainer.test(model, dataloaders=val_dataloader)[0]
    test_result = trainer.test(model, dataloaders=test_dataloader)[0]
    result = {"train": train_result['test_accuracy'],
              "val": val_result['test_accuracy'],
              "test": test_result['test_accuracy']}
    return model, result

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

def print_results(result_dict):
    if "train" in result_dict:
        print(f"Train accuracy: {(100.0*result_dict['train']):4.2f}%")
    if "val" in result_dict:
        print(f"Val accuracy:   {(100.0*result_dict['val']):4.2f}%")
    print(f"Test accuracy:  {(100.0*result_dict['test']):4.2f}%")




if __name__ == "__main__":
    main()
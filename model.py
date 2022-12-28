import torch
from torch import nn
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

class Classifier(nn.Module):
    def __init__(self, embed_dim, h_dim, out_dim, dp_rate=0.1, **kwargs):
        super(Classifier, self).__init__()
        # self.l1 = nn.Linear(in_dim, h_dim)
        # self.tanh1= nn.Tanh()
        # self.dropout = nn.Dropout(dp_rate)
        # self.l2 = nn.Linear(h_dim, out_dim)
        
        self.classify_stack = nn.Sequential(
            nn.Linear(3*embed_dim, h_dim),
            nn.Tanh(),
            nn.Dropout(dp_rate),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Dropout(dp_rate),
            nn.Linear(h_dim, out_dim),
        )
    
    def forward(self, p, h):
        # h = self.l1(x)
        # h = self.tanh1(h)
        # h = self.dropout(h)
        # y = self.l2(h)
        logits = self.classify_stack(torch.cat([p, h, p - h], dim=1))
        return logits

class NLIClassify(pl.LightningModule):

    def __init__(self, batch_size, embed_dim, h_dim, out_dim, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        
        self.batch_size = batch_size
        self.save_hyperparameters()
        # self.d_embed = d_embed
        self.model = Classifier(embed_dim, h_dim, out_dim)
        
        self.loss_module = nn.CrossEntropyLoss()
        # self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=(verb_class+edge_class))
        # self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=(verb_class+edge_class))
        # self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=(verb_class+edge_class))
        
    def forward(self, embeddings, mode='train'):
        p = embeddings['p']
        h = embeddings['h']
        label = embeddings['label']
        
        logits = self.model(p, h)
        loss = self.loss_module(logits, label)
        
        return loss
        
        
             
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        # optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        train_loss = self.forward(batch, mode="train")
        
        self.log("train_loss", train_loss, batch_size=self.batch_size)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        val_loss = self.forward(batch, mode="val")
        
        self.log("val_loss", val_loss, batch_size=self.batch_size)
        return val_loss
import torch
from torch import nn
import torch.optim as optim

import pytorch_lightning as pl
import torchmetrics

class Classifier(nn.Module):
    def __init__(self, embed_dim, h_dim, out_dim, dp_rate=0.1, **kwargs):
        super(Classifier, self).__init__()
        
        self.classify_stack = nn.Sequential(
            nn.Linear(3*embed_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(h_dim, out_dim),
        )
    
    def forward(self, p, h):
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
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=(out_dim))
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=(out_dim))
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=(out_dim))
        
    def forward(self, embeddings, mode='train'):
        p = embeddings['p']
        h = embeddings['h']
        label = embeddings['label']
        
        logits = self.model(p, h)
        loss = self.loss_module(logits, label)
   
        return loss, logits, label
        
        
             
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        # optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        train_loss, logits, label= self.forward(batch, mode="train")
        
        train_pred = logits.argmax(dim=-1)
        train_maj_label = label.argmax(dim=-1)
        
        self.train_acc.update(train_pred, train_maj_label)
        
        self.log("train_loss", train_loss, batch_size=self.batch_size)
        return train_loss
    
    def training_epoch_end(self, training_step_outputs):
        train_accuracy = self.train_acc.compute()
        self.log("train_accuracy", train_accuracy, batch_size=self.batch_size)
        self.train_acc.reset()
        
    def validation_step(self, batch, batch_idx):
        val_loss, logits, label = self.forward(batch, mode="val")
        
        val_pred = logits.argmax(dim=-1)
        val_maj_label = label.argmax(dim=-1)
        
        self.val_acc.update(val_pred, val_maj_label)
        
        self.log("val_loss", val_loss, batch_size=self.batch_size)
        return val_loss
    
    def validation_epoch_end(self, validation_step_outputs):
        val_accuracy = self.val_acc.compute()
        self.log("val_accuracy", val_accuracy, batch_size=self.batch_size)
        self.val_acc.reset()
        
    def test_step(self, batch, batch_idx):
        test_loss, logits, label = self.forward(batch, mode="test")
        
        test_pred = logits.argmax(dim=-1)
        test_maj_label = label.argmax(dim=-1)
        
        self.test_acc.update(test_pred, test_maj_label)
        
        self.log("test_loss", test_loss, batch_size=self.batch_size)
        return test_loss
    
    def test_epoch_end(self, test_step_outputs):
        test_accuracy = self.test_acc.compute()
        self.log("test_accuracy", test_accuracy, batch_size=self.batch_size)
        self.test_acc.reset()
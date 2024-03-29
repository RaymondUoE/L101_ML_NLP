from sentence_transformers.evaluation import LabelAccuracyEvaluator

import torch
from torch.utils.data import DataLoader
import logging
from sentence_transformers.util import batch_to_device
import os
import csv
import pickle


logger = logging.getLogger(__name__)

class MyLabelAccuracyEvaluator(LabelAccuracyEvaluator):
    def __init__(self, dataloader: DataLoader, name: str = "", softmax_model = None, write_csv: bool = True):
        super(MyLabelAccuracyEvaluator, self).__init__(dataloader, 
                                                    name, 
                                                    softmax_model, 
                                                    write_csv)
        
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        total = 0
        correct = 0

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Evaluation on the "+self.name+" dataset"+out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        for step, batch in enumerate(self.dataloader):
            
            features, labels = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            labels = labels.to(model.device)
            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)

            total += prediction.size(0)
            correct += torch.argmax(prediction, dim=1).eq(torch.argmax(labels, dim=1)).sum().item()
        accuracy = correct/total

        logger.info("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy])
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy])

        return accuracy
    
    def eval_model(self, model, dataloader, output_path):
        model.eval()
        total = 0
        correct = 0
        pred = []
        label_record = []
        dataloader.collate_fn = model.model.smart_batching_collate
        for step, batch in enumerate(dataloader):
            features, labels = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.model.device)
            labels = labels.to(model.model.device)
            with torch.no_grad():
                _, prediction = model(features, labels=None)

            total += prediction.size(0)
            correct += torch.argmax(prediction, dim=1).eq(torch.argmax(labels, dim=1)).sum().item()
            pred.append(prediction)
            label_record.append(labels)
        accuracy = correct/total
        print(accuracy)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        with open(f'{output_path}/predicted.pkl', "wb") as f:
            pred = torch.cat(pred, dim=0).cpu().detach().numpy()
            pickle.dump(pred, f)
            f.close()
        with open(f'{output_path}/labels.pkl', "wb") as f:
            label_record = torch.cat(label_record, dim=0).cpu().detach().numpy()
            pickle.dump(label_record, f)
            f.close()
        return accuracy
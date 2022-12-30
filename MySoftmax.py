import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict, Callable
from sentence_transformers import SentenceTransformer
import logging

from sentence_transformers.losses import SoftmaxLoss

logger = logging.getLogger(__name__)

class MySoftmaxLoss(SoftmaxLoss):
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False,
                 loss_fct: Callable = nn.CrossEntropyLoss()):
        super(MySoftmaxLoss, self).__init__(model, 
                                            sentence_embedding_dimension, 
                                            num_labels, 
                                            concatenation_sent_rep, 
                                            concatenation_sent_difference, 
                                            concatenation_sent_multiplication, 
                                            loss_fct)


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a, rep_b = reps

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)
        
        if labels is not None:
            loss = self.loss_fct(output, labels)
            return loss
        else:
            return reps, output
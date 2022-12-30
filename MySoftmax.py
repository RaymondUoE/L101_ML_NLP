import torch
from torch import nn, Tensor
from typing import Iterable, Dict, Callable
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
                 loss_fct: Callable = nn.CrossEntropyLoss(),
                 round=1,
                 hidden=0):
        super(MySoftmaxLoss, self).__init__(model, 
                                            sentence_embedding_dimension, 
                                            num_labels, 
                                            concatenation_sent_rep, 
                                            concatenation_sent_difference, 
                                            concatenation_sent_multiplication, 
                                            loss_fct)
        self.loss_hard = nn.CrossEntropyLoss()
        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1
        # if round == 2:
        #     if hidden == 0:
        #         raise Exception('Need to provide hidden dimension.')
        #     self.classifier = nn.Sequential(
        #                                     nn.Linear(num_vectors_concatenated * sentence_embedding_dimension, hidden),
        #                                     nn.ReLU(),
        #                                     nn.Linear(hidden, num_labels),
        #                                     )

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
            soft_loss = self.loss_fct(output, labels)
            # hard_label = torch.argmax(labels, dim=1)
            # hard_loss = self.loss_hard(output, hard_label)
            # loss = max(soft_loss, hard_loss)
            loss = soft_loss
            return loss
        else:
            return reps, output

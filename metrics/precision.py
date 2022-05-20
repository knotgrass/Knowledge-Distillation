import torch
from torchmetrics import Precision
from os.path import basename
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


preds  = torch.tensor([2, 0, 2, 1])
target = torch.tensor([1, 1, 2, 0])
precision = Precision(average='macro', num_classes=3)
print(precision(preds, target))

precision = Precision(average='micro')
print(precision(preds, target))


classes = []
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}


imgs = []
pred = []
gt = []
assert len(gt) == len(pred), 'len(gt) != len(pred)'


accu = accuracy_score(y_true=gt, y_pred=pred)
prec = precision_score(y_true=gt, y_pred=pred, pos_label=None, average='weighted')
recall = recall_score(y_true=gt, y_pred=pred, pos_label=None, average='weighted')
f1 = f1_score(y_true=gt, y_pred=pred, pos_label=None, average='weighted')

print(accu)
print(prec)
print(recall)
print(f1)

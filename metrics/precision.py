import torch
from torchmetrics import Precision
preds  = torch.tensor([2, 0, 2, 1])
target = torch.tensor([1, 1, 2, 0])
precision = Precision(average='macro', num_classes=3)
print(precision(preds, target))

precision = Precision(average='micro')
print(precision(preds, target))
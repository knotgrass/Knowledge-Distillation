# import torch.nn as nn

import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch import Tensor

# hyperparameters for KD
__alphas__ = [0.99, 0.95, 0.5, 0.1, 0.05]
__temperatures__ = [20., 10., 8., 6., 4.5, 3., 2., 1.5]


def loss_fn_kd(preds: Tensor, labels: Tensor, teacher_preds: Tensor,
               T: float = 6., alpha: float = 0.1):

    return F.kl_div(F.log_softmax(preds / T, dim=1),
                    F.softmax(teacher_preds / T, dim=1),
                    reduction='mean', log_target=False) * (T * T * alpha) + F.cross_entropy(preds, labels) * (1. - alpha)

# def loss_fn_kd(preds:Tensor, labels:Tensor, teacher_preds:Tensor,
#                T:float, alpha:float):
#     return nn.KLDivLoss()(F.log_softmax(preds / T, dim=1),
#                           F.softmax(teacher_preds / T, dim=1)) * (alpha * T * T) + F.cross_entropy(preds, labels) * (1. - alpha)


class KDLoss(_Loss):
    def __init__(self, T: float, alpha: float, reduction: str = 'batchmean') -> None:
        super().__init__()
        self.temperature = T
        self.alpha = alpha
        self.gamma = T * T * alpha
        self.beta = 1. - alpha
        self.reduction = reduction

    def forward(self, preds: Tensor, labels: Tensor, teacher_preds: Tensor) -> Tensor:
        return F.kl_div(F.log_softmax(preds / self.temperature, dim=1), 
                        F.softmax(teacher_preds / self.temperature, dim=1), 
                        reduction=self.reduction, log_target=False 
                        ) * self.gamma + F.cross_entropy(preds, labels) * self.beta

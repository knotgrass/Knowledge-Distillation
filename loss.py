import torch
import torch.nn as nn
import torch.nn.functional as F
Tensor = torch.Tensor

# hyperparameters for KD
alphas = [0.99, 0.95, 0.5, 0.1, 0.05]
temperatures = [20., 10., 8., 6., 4.5, 3., 2., 1.5]


def loss_fn_kd(preds:Tensor, labels:Tensor, teacher_preds:Tensor, T:float, alpha:float):
    return nn.KLDivLoss()(F.log_softmax(preds / T, dim=1),
                          F.softmax(teacher_preds / T, dim=1)) * (alpha * T * T) + F.cross_entropy(preds, labels) * (1. - alpha)



def loss_function_kd(preds:Tensor, labels:Tensor, teacher_preds:Tensor, T:float, alpha:float):
    return F.kl_div(F.log_softmax(preds / T, dim=1), 
                    F.softmax(teacher_preds / T, dim=1),
                    reduction='mean', log_target=False) * T * T * alpha + F.cross_entropy(preds, labels) * (1. - alpha)
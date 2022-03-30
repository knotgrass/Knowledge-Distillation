import torch
import torch.nn as nn
import torch.nn.functional as F
Tensor = torch.Tensor


def loss_fn_kd(preds, labels, teacher_preds, T, alpha):
    return nn.KLDivLoss()(F.log_softmax(preds / T, dim=1),
                          F.softmax(teacher_preds / T, dim=1)) * (alpha * T * T) + F.cross_entropy(preds, labels) * (1. - alpha)



def distillation_loss(preds:Tensor, labels:Tensor, teacher_preds:Tensor, T, alpha:float):
    return F.kl_div(F.log_softmax(preds / T, dim=1), 
                    F.softmax(teacher_preds / T, dim=1),
                    reduction='batchmean') * T * T * alpha + F.cross_entropy(preds, labels) * (1. - alpha)
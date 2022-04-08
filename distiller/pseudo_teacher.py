import torch
import torch.nn as nn
import numpy as np
from random import randint as uniform

class PseudoTeacher(nn.Module):
    def __init__(self, acc:float, num_classes:int, dataset_size:int) -> None:
        self.acc = acc
        self.num_classes = num_classes
        self.dataset_size = dataset_size
        
        # choose random data which teacher will return wrong class with label
        # size dependen of acc and dataset_size
        # output of teacher and label must difference 
        self.list_fn = np.random.randint(low=0, high=dataset_size, 
                                         size= round(acc * dataset_size))
        # tf = round(acc * dataset_size)
        
    def forward(self, y):
        # y is label of class, isn't data
        if uniform(0, 999) <= 999 * self.acc:
            return y
        else:
            return uniform()
    
    def update(self, newacc:float) -> None:
        self.acc = newacc

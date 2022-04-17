import torch
from torch import Tensor
import numpy as np
import copy

class PseudoTeacher:
    def __init__(self, acc:float=0.9, mean:float=-12.0293, std:float= 4.8868, 
                 dataset_size:int=..., num_classes:int=..., seed=None) -> None:
        
        self.acc = acc
        self.mean = mean
        self.std = std
        self.num_classes = num_classes
        self.dataset_size = dataset_size
        
        # choose random data which teacher will return wrong class with label
        # size dependen of acc and dataset_size
        # output of teacher and label in list_fn must difference 
        np.random.seed(seed)
        self.list_fn = np.random.randint(low=0, high=dataset_size, 
                                         size= round((1 - acc) * dataset_size))
        np.random.seed(None)
        # tf = round(acc * dataset_size)
    
    
    def normal_distribution_class(self, idx:int) -> Tensor:
        """
        create probability distribution of output with index
        """
        
        # x is probability distribution vector of output
        x = torch.normal(mean= self.mean, std=self.std, 
                         size= (1, self.num_classes))
        
        argmax = x.argmax()#; print(argmax)
        if argmax != idx:
            max_T = copy.deepcopy(x[0, argmax])
            x[0, argmax] = x[0, idx]
            x[0, idx] = max_T
        return x
        
    def __call__(self, y:int) -> Tensor:
        # y is label of class, isn't data
        if y in self.list_fn:
            return self.random_fn_class(y)
        else:
            return y
        
    def update(self, newacc:float=0.99, newseed=None) -> None:
        """
        this method use to create new teacher 
        with new probability distribution
        of output 
        NOTE: only use if training progress is complete one epoch
        """
        np.random.seed(newseed)
        self.acc = newacc
        self.list_fn = np.random.randint(
            low = 0, high = self.dataset_size, 
            size= round((1. - newacc) * self.dataset_size)
        )
        np.random.seed(None)
        
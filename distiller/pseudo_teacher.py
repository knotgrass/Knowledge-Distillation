from typing import Any
import numpy as np


def random_fn_class(self, idx) -> int:
    # print(idx, end=' ')
    y = np.random.randint(0, self.num_classes)
    # print(y, end= ' ')
    if y == idx:
        y += 1
        if y == self.num_classes:
            y = 0
        # print(y)
        return y
    else:
        # print(y)
        return y

class PseudoTeacher:
    def __init__(self, acc:float=0.99, dataset_size:int=..., 
                 num_classes:int=..., seed=None) -> None:
        
        self.acc = acc
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
    
    def random_fn_class(self, idx) -> int:
        # print(idx, end=' ')
        y = np.random.randint(0, self.num_classes)
        # print(y, end= ' ')
        if y == idx:
            y += 1
            if y == self.num_classes:
                y = 0
            # print(y)
            return y
        else:
            # print(y)
            return y
        
    def __call__(self, y:Any) -> Any:
        # y is label of class, isn't data
        if y in self.list_fn:
            return self.random_fn_class(y)
        else:
            return y
        
    def update(self, newacc:float=0.99, newseed=None) -> None:
        np.random.seed(newseed)
        self.acc = newacc
        self.list_fn = np.random.randint(
            low = 0, high = self.dataset_size, 
            size= round((1. - newacc) * self.dataset_size)
        )
        np.random.seed(None)
        
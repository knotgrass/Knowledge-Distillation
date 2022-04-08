import numpy as np
import math
from ..data import num_classes


def create_matrix(dataset_size:int, acc:float, precision:float, recall:float) -> np.ndarray:
    # use acc for diagonal of matrix
    # các class sai sẽ tuân theo U(0, 1)
    # precision và recall để tính số lượng class sai này
    # qua mỗi epoch sẽ tạo lại matrix 1 lần theo hướng tăng cấc chỉ số lên
    # tất cả đều random lại
    tp = math.floor(dataset_size * acc) # true positive
    fn = dataset_size - tp
    
    # tính số lượng class sai dự vào fn
    
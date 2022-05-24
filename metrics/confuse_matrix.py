import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pretty_confusion_matrix import pp_matrix, pp_matrix_from_data

# np.set_printoptions(precision=5, suppress= True)
#NOTE https://github.com/DTrimarchi10/confusion_matrix.git

path_csv = r'evaluation.csv'

table = pd.read_csv(path_csv)
# print(table.shape)
gt = table['gt'].tolist()
pred = table['pred'].tolist()
size = len(set(gt))

classes = []

# print(len(classes))
# print(set(gt))
# print(set(pred))

# def matrix(gt, pred):
#     assert len(gt) == len(pred), 'len(gt) != len(pred)'
#     confuse_m = np.zeros((size, size), dtype= np.int16)
    
#     for g, p in zip(gt, pred):
#         confuse_m [p][g] += 1

#     return confuse_m

# confuse_m = matrix(gt, pred)
# df_cm = pd.DataFrame(confuse_m, index=range(1, size+1), columns=range(1, size+1))
# pp_matrix(df_cm, cmap='YlGnBu')

# sns.dark_palette("#69d", reverse=True, as_cmap=True)
pp_matrix_from_data(y_test=gt, predictions=pred, columns=classes, cmap="RdBu")

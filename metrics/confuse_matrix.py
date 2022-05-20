import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pretty_confusion_matrix import pp_matrix, pp_matrix_from_data

# np.set_printoptions(precision=5, suppress= True)


path_csv = r'C:\Users\adang1\Documents\SuntoryImageDB-dev_python\temp\evaluation.csv'

# test = r'C:\Users\adang1\Documents\SuntoryImageDB-dev_python\train_datasets\cls\test_classify'

table = pd.read_csv(path_csv)
# print(table.shape)
gt = table['gt'].tolist()
pred = table['pred'].tolist()
size = len(set(gt))

classes = [
        "TP_1",
        "TP_2",
        "TP_3",
        "TP_4",
        "TP_5",
        "TP_6",
        "TP_7",
        "TP_8",
        "TP_unknown",
        
        "FN_1",
        "FN_2",
        "FN_3",
        "FN_4",
        "FN_5",
        "FN_6",
        "FN_7",
        "FN_unknown"]

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

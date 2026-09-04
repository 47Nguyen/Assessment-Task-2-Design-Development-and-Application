import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

train_path = './A2_FashionDataset/FashionDataset/train/styles_train.csv'
test_path = './A2_FashionDataset/FashionDataset/test/styles_prediction.csv'

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# print(df_train.head())


print(df_test.head())


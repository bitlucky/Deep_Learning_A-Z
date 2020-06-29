# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from keras.util.np_utils import to_categorical

dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[: , 3:13].values
Y = dataset.iloc[: , 13].values

X = to_categorical(X)

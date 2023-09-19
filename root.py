import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


df= pd.read_csv('features.csv')

X= df.iloc[:,:-1].values
Y=df.iloc[:,-1].values

# for production encoder use
prod_Y= Y.copy()

x_train, x_test, y_train, y_test= train_test_split(X, Y, random_state=0, shuffle=True)

# for production Standardscaler use
prod_std= x_train.copy()

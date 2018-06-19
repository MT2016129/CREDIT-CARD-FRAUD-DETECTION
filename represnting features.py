import numpy as np 
import pandas as pd 

import matplotlib
import matplotlib.pyplot as plt
#matplotlib inline

data = pd.read_csv("NonFraudUnderSampled.csv")

from sklearn.preprocessing import StandardScaler

##data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
##data.head()
##data = data.drop(['Time','Amount'],axis=1)
##
##normal_data = data.loc[data["Class"] == 0]
##fraud_data = data.loc[data["Class"] == 1]
##print("Size of normal_data:", len(normal_data))
##print("Size of fraud_data:", len(fraud_data))

matplotlib.style.use('ggplot')
pca_columns = list(data)[0:-2]
datald = (data[pca_columns]).iloc[:,0:4]
datald.hist(stacked=False, bins=100, figsize=(10,30), layout=(2,2))
plt.show()
print ("SHRADDHA")

##matplotlib.style.use('ggplot')
##pca_columns = list(data)[0:-2]
##fraudd = (fraud_data[pca_columns]).iloc[:,0:4]
##fraudd.hist(stacked=False, bins=100, figsize=(10,20), layout=(2,2))
##plt.show()

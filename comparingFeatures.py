import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

from sklearn.preprocessing import StandardScaler
dataOrg = pd.read_csv("creditcard.csv")
#dataOrg.head()

#normalizing the amount column and droping the time column
dataOrg['normAmount'] = StandardScaler().fit_transform(dataOrg['Amount'].reshape(-1, 1))
dataOrg = dataOrg.drop(['Time','Amount'],axis=1)
dataOrg.head()


#count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
#dropping all the fraud data from the dataset 
data = dataOrg[dataOrg.Class != 1]
datafraud = dataOrg[dataOrg.Class != 0]


a1=np.array(data.V1)
a2=np.array(data.V2)
a3=np.array(data.V3)
a4=np.array(data.V4)
a5=np.array(data.V5)
a6=np.array(data.V6)
a7=np.array(data.V7)
a8=np.array(data.V8)
a9=np.array(data.V9)
a10=np.array(data.V10)
a11=np.array(data.V11)
a12=np.array(data.V12)
a13=np.array(data.V13)
a14=np.array(data.V14)
a15=np.array(data.V15)
a16=np.array(data.V16)
a17=np.array(data.V17)
a18=np.array(data.V18)
a19=np.array(data.V19)
a20=np.array(data.V20)
a21=np.array(data.V21)
a22=np.array(data.V22)
a23=np.array(data.V23)
a24=np.array(data.V24)
a25=np.array(data.V25)
a26=np.array(data.V26)
a27=np.array(data.V27)
a28=np.array(data.V28)
a29=np.array(data.normAmount)


b1=np.array(datafraud.V1)
b2=np.array(datafraud.V2)
b3=np.array(datafraud.V3)
b4=np.array(datafraud.V4)
b5=np.array(datafraud.V5)
b6=np.array(datafraud.V6)
b7=np.array(datafraud.V7)
b8=np.array(datafraud.V8)
b9=np.array(datafraud.V9)
b10=np.array(datafraud.V10)
b11=np.array(datafraud.V11)
b12=np.array(datafraud.V12)
b13=np.array(datafraud.V13)
b14=np.array(datafraud.V14)
b15=np.array(datafraud.V15)
b16=np.array(datafraud.V16)
b17=np.array(datafraud.V17)
b18=np.array(datafraud.V18)
b19=np.array(datafraud.V19)
b20=np.array(datafraud.V20)
b21=np.array(datafraud.V21)
b22=np.array(datafraud.V22)
b23=np.array(datafraud.V23)
b24=np.array(datafraud.V24)
b25=np.array(datafraud.V25)
b26=np.array(datafraud.V26)
b27=np.array(datafraud.V27)
b28=np.array(datafraud.V28)
b29=np.array(datafraud.normAmount)

import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(5, 5)

for j in range(29):
    x1='a'+str(j+1)
    x2='b'+str(j+1)
    plt.figure("COMPARISION OF V"+str(j))
    for i in range(29):
        plt.subplot(5,6,i+1)
        t1='a'+str(i+1)
        t2='b'+str(i+1)
        Z = np.column_stack((eval(x1),eval(t1)))
        D = np.column_stack((eval(x2),eval(t2)))
        plt.scatter(Z[:,0],Z[:,1],c = 'y')
        plt.scatter(D[:,0],D[:,1],c = 'k')
        plt.title("V"+str(j+1)+" & V"+str(i+1))
        plt.xlabel('X'),plt.ylabel("Y")
        plt.axis('off')
    plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()


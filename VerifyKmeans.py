import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
import csv
import random

#getting the data from CSV file
dataOrg = pd.read_csv("creditcard.csv")

#normalizing the amount column and droping the time column
dataOrg['normAmount'] = StandardScaler().fit_transform(dataOrg['Amount'].reshape(-1, 1))
dataOrg = dataOrg.drop(['Time','Amount'],axis=1)
dataOrg.head()

#separating raud and Non Fraud data
dataNonfraud = dataOrg[dataOrg.Class != 1]
datafraud = dataOrg[dataOrg.Class != 0]
datafraud = datafraud.drop(['Class'],axis=1)
dataNonfraud = dataNonfraud.drop(['Class'],axis=1)


#Taking each feature in array format of Non Fraud data so that we can apply k means clustering into that
a1=np.array(dataNonfraud.V1)
a2=np.array(dataNonfraud.V2)
a3=np.array(dataNonfraud.V3)
a4=np.array(dataNonfraud.V4)
a5=np.array(dataNonfraud.V5)
a6=np.array(dataNonfraud.V6)
a7=np.array(dataNonfraud.V7)
a8=np.array(dataNonfraud.V8)
a9=np.array(dataNonfraud.V9)
a10=np.array(dataNonfraud.V10)
a11=np.array(dataNonfraud.V11)
a12=np.array(dataNonfraud.V12)
a13=np.array(dataNonfraud.V13)
a14=np.array(dataNonfraud.V14)
a15=np.array(dataNonfraud.V15)
a16=np.array(dataNonfraud.V16)
a17=np.array(dataNonfraud.V17)
a18=np.array(dataNonfraud.V18)
a19=np.array(dataNonfraud.V19)
a20=np.array(dataNonfraud.V20)
a21=np.array(dataNonfraud.V21)
a22=np.array(dataNonfraud.V22)
a23=np.array(dataNonfraud.V23)
a24=np.array(dataNonfraud.V24)
a25=np.array(dataNonfraud.V25)
a26=np.array(dataNonfraud.V26)
a27=np.array(dataNonfraud.V27)
a28=np.array(dataNonfraud.V28)
a29=np.array(dataNonfraud.normAmount)

#merging all the features, appending them into a stack
Z = np.column_stack((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29))
Z = np.float32(Z)

NC=3
# define criteria and apply kmeans() on Non Fraud data
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
ret,label,center=cv2.kmeans(Z,NC,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# number of Clusters we want is __ for simplicity it is hard coded

A = Z[label.ravel()==0]
B = Z[label.ravel()==1]
C = Z[label.ravel()==2]
##D = Z[label.ravel()==3]
##E = Z[label.ravel()==4]
##F = Z[label.ravel()==5]
##G = Z[label.ravel()==6]
##H = Z[label.ravel()==7]
##I = Z[label.ravel()==8]
##J = Z[label.ravel()==9]

#printing Each cluster sizes :
print ("CLUSTER SIZES ARE :")
for i in range(NC):
    t=chr(65+i)
    print ("cluster " +str( chr(65+i))+": "+str(len(eval(t))))

totalSize =len(Z)
print ("\nTotal Size of Non Fraud Data is "+str(totalSize))


fraudlen = len(datafraud) * 1 #HERE# WE CAN SoME WHAT INCREASE THE SIZE OF NON FRAUD DATA WHILE UNDERSAMPLING ALSO NO NEED TO KEEP 50-50
#getting number of data points to be picked fron each cluster
weighA = int((len(A)/totalSize) * fraudlen)
weighB = int((len(B)/totalSize) * fraudlen)
weighC = int((len(C)/totalSize) * fraudlen)
##weighD = int((len(D)/totalSize) * fraudlen)
##weighE = int((len(E)/totalSize) * fraudlen)
##weighF = int((len(F)/totalSize) * fraudlen)
##weighG = int((len(G)/totalSize) * fraudlen)
##weighH = int((len(H)/totalSize) * fraudlen)
##weighI = int((len(I)/totalSize) * fraudlen)
##weighJ = int((len(J)/totalSize) * fraudlen)


print ("\nNumber of elements to pick from each clusters")
for i in range(NC):
    t='weigh'+chr(65+i)
    print ("cluster " +str( chr(65+i))+": "+str(eval(t)))

#Now we know the number of data to be picked from each cluster so, we randomly pick same number of data from each.
import random
selectedA = random.sample(list(A), k=weighA)
selectedB = random.sample(list(B), k=weighB)
selectedC = random.sample(list(C), k=weighC)
##selectedD = random.sample(list(D), k=weighD)
##selectedE = random.sample(list(E), k=weighE)
##selectedF = random.sample(list(F), k=weighF)
##selectedG = random.sample(list(G), k=weighG)
##selectedH = random.sample(list(H), k=weighH)
##selectedI = random.sample(list(I), k=weighI)
##selectedJ = random.sample(list(J), k=weighJ)

#chaining the selected sample into one for storing it in csv file
dataSampled = selectedA
import itertools
for i in range(NC-1):
    t= 'selected'+chr(66+i)
    dataSampled = itertools.chain(dataSampled,eval(t))
dataSampled = list(dataSampled)

import numpy as np

print ("\nVariance of the sampled data")
print (np.var(selectedA))
print (np.var(selectedB))
print (np.var(selectedC))
print ("\nvaraiance of the population")

print (np.var(A))
print (np.var(B))
print (np.var(C))

print ("\n Mean if the actual clusters")
print (np.mean(A))
print (np.mean(B))
print (np.mean(C))


##
###header of the csv file 
##head = "V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, V29"
##np.savetxt("NonFraudUnderSampled_Kmeans.csv", dataSampled, delimiter=",", header=head,comments='')
##np.savetxt("Fraud.csv",datafraud,delimiter=",", header=head,comments='')
##
##lines = dataNonfraud.values.tolist()
##selectedRandomly = random.sample(lines, fraudlen)
##np.savetxt("NonFraudUnderSampled.csv", selectedRandomly, delimiter=",", header=head,comments='')
##
##
###merging the fraud and non-fraud undersampled into one file
##a = pd.read_csv("Fraud.csv")
##b = pd.read_csv("NonFraudUnderSampled_Kmeans.csv")
##a['Class']=1
##b['Class']=0
##
##undersampled_data = pd.concat([a, b], join='outer')
##undersampled_data.to_csv("KmeanComplete.csv", index=False)
##
##b = pd.read_csv("NonFraudUnderSampled.csv")
##a['Class']=1
##b['Class']=0
##
##undersampled_data = pd.concat([a, b], join='outer')
##undersampled_data.to_csv("UnderSampledComplete.csv", index=False)
##

cv2.waitKey(0)
cv2.destroyAllWindows()

#Dont remeber Where I used it 
###doing the Same with fraudulent data
##b1=np.array(datafraud.V1)
##b2=np.array(datafraud.V2)
##b3=np.array(datafraud.V3)
##b4=np.array(datafraud.V4)
##b5=np.array(datafraud.V5)
##b6=np.array(datafraud.V6)
##b7=np.array(datafraud.V7)
##b8=np.array(datafraud.V8)
##b9=np.array(datafraud.V9)
##b10=np.array(datafraud.V10)
##b11=np.array(datafraud.V11)
##b12=np.array(datafraud.V12)
##b13=np.array(datafraud.V13)
##b14=np.array(datafraud.V14)
##b15=np.array(datafraud.V15)
##b16=np.array(datafraud.V16)
##b17=np.array(datafraud.V17)
##b18=np.array(datafraud.V18)
##b19=np.array(datafraud.V19)
##b20=np.array(datafraud.V20)
##b21=np.array(datafraud.V21)
##b22=np.array(datafraud.V22)
##b23=np.array(datafraud.V23)
##b24=np.array(datafraud.V24)
##b25=np.array(datafraud.V25)
##b26=np.array(datafraud.V26)
##b27=np.array(datafraud.V27)
##b28=np.array(datafraud.V28)
##b29=np.array(datafraud.normAmount)
##X = np.column_stack((b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,b25,b26,b27,b28,b29))

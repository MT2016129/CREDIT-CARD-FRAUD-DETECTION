#Linear Regression on data
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')

#    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def show_data(cm, print_res = 0):
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    tn = cm[0,0]
    if print_res == 1:
        print('Precision =     {:.3f}'.format(tp/(tp+fp)))
        print('Recall (TPR) =  {:.3f}'.format(tp/(tp+fn)))
        print('Fallout (FPR) = {:.3e}'.format(fp/(fp+tn)))
    return tp/(tp+fp), tp/(tp+fn), fp/(fp+tn)





a = pd.read_csv("FraudSVM.csv")
b = pd.read_csv("foo.csv")
undersampled_data = pd.concat([a, b], join='outer')
undersampled_data.to_csv("outputLR.csv", index=False)
print ("checking if there is any missing values ")
undersampled_data.isnull().any()

#count_Class=pd.value_counts(undersampled_data["Class"], sort= True)
#count_Class.plot(kind= 'bar')

X= undersampled_data.iloc[:, undersampled_data.columns != "Class"].values

y= undersampled_data.iloc[:, undersampled_data.columns == "Class"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 0)
print("The split of the under_sampled data is as follows")
print("X_train: ", len(X_train))
print("X_test: ", len(X_test))
print("y_train: ", len(y_train))
print("y_test: ", len(y_test))

lrn = LogisticRegression()

skf = StratifiedKFold(n_splits = 5, shuffle = True)
##for train_index, test_index in skf.split(X, y):
##    X_train, y_train = X[train_index], y[train_index]
##    X_test, y_test = X[test_index], y[test_index]
##    break

lrn.fit(X_train, y_train)
y_pred = lrn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
if lrn.classes_[0] == 1:
    cm = np.array([[cm[1,1], cm[1,0]], [cm[0,1], cm[0,0]]])

plot_confusion_matrix(cm, ['0', '1'], )
pr, tpr, fpr = show_data(cm, print_res = 1)




dataOrg = pd.read_csv("creditcard.csv")
#dataOrg.head()
dataOrg['normAmount'] = StandardScaler().fit_transform(dataOrg['Amount'].reshape(-1, 1))
dataOrg = dataOrg.drop(['Time','Amount'],axis=1)

X= dataOrg.iloc[:, dataOrg.columns != "Class"].values
y= dataOrg.iloc[:, dataOrg.columns == "Class"].values

y_pred = lrn.predict(X)
cm = confusion_matrix(y, y_pred)
if lrn.classes_[0] == 1:
    cm = np.array([[cm[1,1], cm[1,0]], [cm[0,1], cm[0,0]]])

plot_confusion_matrix(cm, ['0', '1'], )
pr, tpr, fpr = show_data(cm, print_res = 1)



#undersampling randomly
#count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
#dropping all the fraud data from the dataset
X = dataOrg.ix[:, dataOrg.columns != 'Class']
y = dataOrg.ix[:, dataOrg.columns == 'Class']

# Number of data points in the minority class
number_records_fraud = len(dataOrg[dataOrg.Class == 1])
fraud_indices = np.array(dataOrg[dataOrg.Class == 1].index)

# Picking the indices of the normal classes
normal_indices = dataOrg[dataOrg.Class == 0].index

# Out of the indices we picked, randomly select "x" number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)

# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

# Under sample dataset
under_sample_data = dataOrg.iloc[under_sample_indices,:]

X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

# Showing ratio
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))

##data = dataOrg[dataOrg.Class != 1]
##datafraud = dataOrg[dataOrg.Class != 0]
##
##from sklearn.preprocessing import StandardScaler
##
##weighA = (len(datafraud))
##print(weighA)
##print (len(data))
##import random
##selectedA = random.sample(list(data), k=weighA)
##print (selectedA)

X_train, X_test, y_train, y_test = train_test_split(X_undersample, y_undersample, test_size= 0.25, random_state= 0)
##print("The split of the under_sampled data is as follows")
##print("X_train: ", len(X_train))
##print("X_test: ", len(X_test))
##print("y_train: ", len(y_train))
##print("y_test: ", len(y_test))

lrn = LogisticRegression()

skf = StratifiedKFold(n_splits = 5, shuffle = True)
##for train_index, test_index in skf.split(X, y):
##    X_train, y_train = X[train_index], y[train_index]
##    X_test, y_test = X[test_index], y[test_index]
##    break

lrn.fit(X_train, y_train)
y_pred = lrn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
if lrn.classes_[0] == 1:
    cm = np.array([[cm[1,1], cm[1,0]], [cm[0,1], cm[0,0]]])

plot_confusion_matrix(cm, ['0', '1'], )
pr, tpr, fpr = show_data(cm, print_res = 1)

X= dataOrg.iloc[:, dataOrg.columns != "Class"].values
y= dataOrg.iloc[:, dataOrg.columns == "Class"].values

y_pred = lrn.predict(X)
cm = confusion_matrix(y, y_pred)
if lrn.classes_[0] == 1:
    cm = np.array([[cm[1,1], cm[1,0]], [cm[0,1], cm[0,0]]])

plot_confusion_matrix(cm, ['0', '1'], )
pr, tpr, fpr = show_data(cm, print_res = 1)

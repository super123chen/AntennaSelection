import math
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict,KFold
from time import  time
from numpy import mat


x = pd.read_csv(open(r'/home/wwj/chenqiliang/8822_list.csv'), header=None)
matrix = x.values
def GetIndexFrom(y_pre):
    for i in range(0, 784):
        if y_pre == matrix[i][0]:
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i,4]


m = n = 8
I = np.eye(n)
dataset = pd.read_csv(open(r'/home/wwj/chenqiliang/SVD/All-channel_matrix_p_1.csv')).iloc[:200000, 1:]
label_array = pd.read_csv(open(r'/home/wwj/chenqiliang/SVD/gain_labels_p_1.csv')).iloc[:200000, 1:]

a = 10

knn = KNeighborsClassifier()
kf = KFold(n_splits=5,shuffle=True)


Loss_Gain = []          
KNN_Score = []          
Predict_Gain = []       
trainTime = []              
testTime = []               

for train_index, test_index in kf.split(dataset):
    xTrain, xTest = dataset.iloc[train_index], dataset.iloc[test_index]
    yTrain, yTest = label_array.iloc[train_index], label_array.iloc[test_index]

    trainStart = time()
    knn.fit(xTrain,yTrain)                    
    train = time() - trainStart
    trainTime.append(train)
    print('trainTime', trainTime)

    testStart = time()
    predict = knn.predict(xTest)             
    test = time() - testStart
    testTime.append(test)
    print('testTime',testTime)

    
    KNN_Score.append(knn.score(xTest, yTest))

    yPredict_np = np.array(predict)    
    yTest_np = np.array(yTest)         
    xTest_np = np.array(xTest)         

    
    for i in range(40000):
        C= []
        ArrayA = xTest_np[i].reshape(8, 8)
        ArrayA = np.matrix(ArrayA)

        KNN_i1, KNN_i2, KNN_j1, KNN_j2 = GetIndexFrom(yPredict_np[i])    
        KNN_Sub = mat(np.zeros((2, 2)), dtype=float)
        KNN_Sub[0, 0] = ArrayA[KNN_i1, KNN_j1]
        KNN_Sub[0, 1] = ArrayA[KNN_i1, KNN_j2]
        KNN_Sub[1, 0] = ArrayA[KNN_i2,KNN_j1]
        KNN_Sub[1, 1] = ArrayA[KNN_i2, KNN_j2]
        Full_Gain = math.sqrt(1 / 2) * np.linalg.norm(ArrayA, ord='fro')
        Sub_Gain = math.sqrt(1 / 2) * np.linalg.norm(KNN_Sub, ord='fro')

        Predict_Gain.append(Sub_Gain)
        C.append(Full_Gain - Sub_Gain)
    Loss_Gain.append(np.mean(C))
    print(Loss_Gain)
accuracy = np.mean(KNN_Score)
Gain_Mean = np.mean(Predict_Gain)
Loss_Mean = np.mean(Loss_Gain)
Loss_Variance = np.var(Loss_Gain)


train = np.mean(trainTime)
test = np.mean(testTime)
if test < 1e-6:
    testUnit = "ns"
    test *= 1e9
elif test < 1e-3:
    testUnit = "us"
    test *= 1e6
elif test < 1:
    testUnit = "ms"
    test *= 1e3
else:
    testUnit = "s"

if train < 1e-6:
    trainUnit = "ns"
    train *= 1e9
elif train < 1e-3:
    trainUnit = "us"
    train *= 1e6
elif train < 1:
    trainUnit = "ms"
    train *= 1e3
else:
    trainUnit = "s"











print("160000 train time %.1f %s" % (train, trainUnit))
print("40000 test time %.1f %s" % (test, testUnit))
print('acc', accuracy)
print('sub gain mean', Gain_Mean)
print('loss mean', Loss_Mean)
print('loss variance', Loss_Variance)

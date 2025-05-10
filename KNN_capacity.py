



import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from time import time
from numpy import mat


x = pd.read_csv(open(r'/home/wwj/chenqiliang/8822_list.csv'), header=None)
matrix = x.values
def GetIndexFrom(y_pre):
    for i in range(0, 784):
        if y_pre == matrix[i][0]:
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i,4]



a = 100

m = n = 8


dataset = pd.read_csv(open(r'/home/wwj/chenqiliang/SVD/All-channel_matrix_p_20.csv')).iloc[:200000, 1:]
label_array = pd.read_csv(open(r'/home/wwj/chenqiliang/SVD/capacity_labels_p_20.csv')).iloc[:200000, 1:]

knn = KNeighborsClassifier()
kf = KFold(n_splits=5,shuffle=True)


Loss_Capacity = []          
KNN_Score = []              
Predict_Capacity = []       
trainTime = []              
testTime = []               

I1 = np.eye(8)
I2 = np.eye(2)

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
    print('testTime', testTime)

    yPredict_np = np.array(predict)      
    yTrue_np = np.array(yTest)           
    xTest_np = np.array(xTest)           

    same = 0
    for i in range(0, len(yTrue_np)):
        if yTrue_np[i] == yPredict_np[i]:
            same = same + 1
    KNN_Score.append(same / len(yTrue_np))
    print(KNN_Score)

    
    for i in range(40000):
        ArrayA = xTest_np[i].reshape(8, 8)
        ArrayA = np.matrix(ArrayA)

        KNN_i1, KNN_i2, KNN_j1, KNN_j2 = GetIndexFrom(yPredict_np[i])    
        KNN_Sub = mat(np.zeros((2, 2)), dtype=float)
        KNN_Sub[0, 0] = ArrayA[KNN_i1, KNN_j1]
        KNN_Sub[0, 1] = ArrayA[KNN_i1, KNN_j2]
        KNN_Sub[1, 0] = ArrayA[KNN_i2, KNN_j1]
        KNN_Sub[1, 1] = ArrayA[KNN_i2, KNN_j2]
        Full_Capacity  =  np.log2(np.linalg.det(I1 + a * ArrayA.T * ArrayA / 8))
        Sub_Capacity  = np.log2(np.linalg.det(I2 + a * KNN_Sub.T * KNN_Sub / 2))

        Predict_Capacity.append(Sub_Capacity)
        Loss_Capacity.append(Full_Capacity  - Sub_Capacity)
    print('Loss_Capacity: ',np.mean(Loss_Capacity))


accuracy = np.mean(KNN_Score)
Predict_Capacity_Mean = np.mean(Predict_Capacity)
Loss_Mean = np.mean(Loss_Capacity)
Loss_Variance = np.var(Loss_Capacity)


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
















print("SNR", a)

print("160000 train time %.1f %s" % (train, trainUnit))
print("40000 test time %.1f %s" % (test, testUnit))









print('预测准确率', accuracy)
print('预测子信道容量均值', Predict_Capacity_Mean)




print('预测损失均值', Loss_Mean)



















def calculate_ber(predicted, true):
    
    errors = np.sum(predicted != true) 
    total_bits = predicted.size
    return errors / total_bits  


accuracy = np.mean(KNN_Score)
ber = calculate_ber(yPredict_np, yTrue_np)


print(f"预测准确率: {accuracy:.6f}")
print(f"比特误率 (BER): {ber:.6f}")





print('预测损失方差', Loss_Variance)

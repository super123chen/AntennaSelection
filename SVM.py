
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,cross_val_predict,KFold
from time import  time
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

#分类器
svm = SVC()
kf = KFold(n_splits=5,shuffle=True)
I1 = np.eye(8)
I2 = np.eye(2)

Loss_Capacity = []          #5次 容量损失列表(全信道容量-预测子信道容量)   输出结果求平均
SVM_Score = []              #5次 准确率列表  输出结果求平均
Predict_Capacity = []       #10w(5折*20000)预测子信道增益列表   输出结果求平均
trainTime = []              #5次 训练时间列表 输出结果求平均
testTime = []               #5次 预测时间列表 输出结果求平均
for train_index, test_index in kf.split(dataset):
    xTrain, xTest = dataset.iloc[train_index], dataset.iloc[test_index]
    yTrain, yTest = label_array.iloc[train_index], label_array.iloc[test_index]

    trainStart = time()
    svm.fit(xTrain,yTrain)
    train = time() - trainStart
    trainTime.append(train)
    print('trainTime', trainTime)

    testStart = time()
    predict = svm.predict(xTest)
    test = time() - testStart
    testTime.append(test)
    print('testTime', testTime)

    yPredict_np = np.array(predict)   #预测的y值
    yTrue_np = np.array(yTest)        #实际的y值
    xTest_np = np.array(xTest)        #实际的x值
    same = 0
    for i in range(0, len(yTrue_np)):
        if (yTrue_np[i] == yPredict_np[i]):
            same = same + 1
    SVM_Score.append(same / len(yTrue_np))
    print(SVM_Score)

    #我们用的是1000个数
    # 据5折，所以每折就200个
    for i in range(40000):
        C = []
        ArrayA = xTest_np[i].reshape(8, 8)
        ArrayA = np.matrix(ArrayA)

        SVM_i1, SVM_i2, SVM_j1, SVM_j2 = GetIndexFrom(yPredict_np[i])
        SVM_Sel_B = mat(np.zeros((2, 2)), dtype=float)
        SVM_Sel_B[0, 0] = ArrayA[SVM_i1, SVM_j1]
        SVM_Sel_B[0, 1] = ArrayA[SVM_i1, SVM_j2]
        SVM_Sel_B[1, 0] = ArrayA[SVM_i2, SVM_j1]
        SVM_Sel_B[1, 1] = ArrayA[SVM_i2, SVM_j2]

        Full_capacity = np.log2(np.linalg.det(I1 + a * ArrayA.T * ArrayA / 8))
        Sub_capacity = np.log2(np.linalg.det(I2 + a * SVM_Sel_B.T * SVM_Sel_B / 2))
        Predict_Capacity.append(Sub_capacity)
        C.append(Full_capacity - Sub_capacity)
    Loss_Capacity.append(np.mean(C))
    print(Loss_Capacity)

accuracy = np.mean(SVM_Score)
Capacity_Mean = np.mean(Predict_Capacity)
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




print("基于信道容量，信噪比",a)
print("SVM(200000个测试样本)")
print("160000个样本的训练时间 %.1f %s" % (train, trainUnit))
print("40000个样本的测试时间 %.1f %s" % (test, testUnit))




print('预测准确率', accuracy)
print('预测子信道容量均值', Capacity_Mean)



#没用到
print('预测损失均值', Loss_Mean)
#没用到
print('预测损失方差', Loss_Variance)

from __future__ import division, print_function, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from time import time
from numpy import mat
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import torch.optim as optim
from torch.optim import AdamW



x = pd.read_csv(r'/home/wwj/chenqiliang/8822_list.csv', header=None)
matrix = x.values
def GetIndexFrom(y_pre):
    for i in range(0, 784):
        if y_pre == matrix[i][0]:
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i,4]



a = 50
n = 8
X_orig = pd.read_csv(r"/home/wwj/chenqiliang/wubiaoqian/All-channel_matrix_p_50.csv").iloc[:, 1:]
X_orig = np.asarray(X_orig, np.float32)
dataset = X_orig.reshape(X_orig.shape[0],8,8,1)

label = pd.read_csv(r"/home/wwj/chenqiliang/wubiaoqian/capacity_labels_p_50.csv").iloc[:, 1]
label = np.asarray(label, np.int32)
label.astype(np.int32)
n_class =  784



n_sample = label.shape[0]
label_array = np.zeros((n_sample, n_class))   
for i in range(n_sample):
    label_array[i, label[i] - 1] = 1  


xTrain, xTest, yTrain, yTest = train_test_split(dataset, label_array, test_size=0.2, random_state=40)
print("xTrain: ",len(xTrain))
print("xTest: ",len(xTest))




class CNN(nn.Module):
    def __init__(self,n_class):
        super(CNN, self).__init__()
        
        self.features = nn.Sequential(
           
           
            nn.Conv2d(in_channels=1,out_channels=96,kernel_size=5,stride=2),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=3,padding=2),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=256,out_channels=384,padding=1,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
       
        self.classifier = nn.Sequential(
            
           
            nn.Linear(in_features=2*2*256,out_features=4096),
            nn.ReLU(),
            
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
           
            nn.Linear(4096,784)
        )

   
    def forward(self,x):
        x = x.reshape(-1, 1, 8, 8)
        x = self.features(x)
       
        x = torch.flatten(x,1)
        result = self.classifier(x)
        return result









n_class = 784  
model = CNN(n_class)













train_dataset = TensorDataset(torch.Tensor(xTrain), torch.Tensor(yTrain))
test_dataset = TensorDataset(torch.Tensor(xTest), torch.Tensor(yTest))

train_loader = DataLoader(train_dataset, batch_size=1280, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1280, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = CNN(784).to(device)
#model.load_state_dict(torch.load(f"/home/wwj/chenqiliang/model.pth"))





#
lr = 0.0001 
initial_lr=0.1
weight_decay = 0.0001  
betas = (0.95, 0.999)  
#eps = 1e-8
eps=1e-8
momentum=0.9
alpha=0.99
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
#loss_fn = nn.CrossEntropyLoss()



criterion = nn.CrossEntropyLoss()
current_lr = optimizer.param_groups[0]['lr']
best_loss = float('inf') 


trainStart = time()
num_epochs = 100



for epoch in range(num_epochs):
    model.train()
    t = tqdm(train_loader, total=len(train_loader))

    for inputs, labels in t:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = torch.argmax(labels, dim=1)
        

        optimizer.zero_grad()
     

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
    if loss < best_loss:
        best_loss = loss.item()
        
        save_file_name = f"/home/wwj/chenqiliang/model.pth"  
        torch.save(model.state_dict(), save_file_name)

    print("epoch is {},loss is {}".format(epoch, loss))


train = time() - trainStart






'''
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)

trainStart = time()
model.fit(xTrain, yTrain, n_epoch=100, validation_set=(xTest, yTest), shuffle=True,
          show_metric=True, batch_size=1280, snapshot_step=200,
          snapshot_epoch=False, run_id='alexnet_oxflowers17')
          
'''          
          
          
          





model = CNN(784)
model.load_state_dict(torch.load(f"/home/wwj/chenqiliang/model.pth"))
device = torch.device('cpu')
model.to(device)

ResNet_Pre1 = model(torch.from_numpy(xTest[0:40000]).to(device))


########################################################333wo xiede
testStart=time()
pre_array1 = np.zeros((40000, n_class))
index1=torch.argmax(ResNet_Pre1,dim=1)
for i in range(40000):
    pre_array1[i,index1[i]] = 1
    
    
test = time() - testStart   

aaa1 = torch.argmax(ResNet_Pre1, axis=1) + 1
ResNet_Pre_np1 = aaa1.numpy()
ResNet_Pre_np1_ = pre_array1
ResNet_Pre_np_=ResNet_Pre_np1_



xTest_np = np.array(xTest[0:40000])




label_array = np.zeros((n_sample, n_class))
for i in range(n_sample):
    label_array[i, label[i] - 1] = 1

yTest_indices = np.array(yTest[:40000])



b=np.all(ResNet_Pre_np_ == yTest_indices, axis=1)

acc = np.sum(b) / 40000.0 * 100.0







fullCapacity = pd.read_csv(r"/home/wwj/chenqiliang/wubiaoqian/All-channel_matrix_p_50.csv").iloc[:, 1:]
fullCapacity = fullCapacity[0:200000]
fullCapacity  = np.asarray(fullCapacity, np.float32)
Best_subCapacity = pd.read_csv(r"/home/wwj/chenqiliang/wubiaoqian/Sub_channel_capacity_p_50.csv").iloc[:, 1:]
Best_subCapacity = Best_subCapacity[0:200000]
Best_subCapacity = np.asarray(Best_subCapacity, np.float32)

fullCapacity_Mean = np.mean(fullCapacity)

Best_subCapacity_Mean = np.mean(Best_subCapacity)


I1 = np.eye(8)
I2 = np.eye(2)
Pre_Loss = []
Pre_Capacity = []
for i in range(40000):
    ArrayA = xTest_np[i].reshape(8, 8)
    ArrayA = np.matrix(ArrayA)

    i1, i2, j1, j2 = GetIndexFrom(ResNet_Pre_np1[i])
    Pre_sub = mat(np.zeros((2, 2)), dtype=float)
    Pre_sub[0, 0] = ArrayA[i1, j1]
    Pre_sub[0, 1] = ArrayA[i1, j2]
    Pre_sub[1, 0] = ArrayA[i2, j1]
    Pre_sub[1, 1] = ArrayA[i2, j2]
    Pre_fullCapacity =  np.log2(np.linalg.det(I1 + a * ArrayA.T * ArrayA / 8))
    Pre_subCapacity = np.log2(np.linalg.det(I2 + a * Pre_sub.T * Pre_sub / 2))
    Pre_Capacity.append(Pre_subCapacity)
    Pre_Loss.append(Pre_fullCapacity - Pre_subCapacity)


Capacity_Mean = np.mean(Pre_Capacity)
Loss_Mean = np.mean(Pre_Loss)
Loss_Variance = np.var(Pre_Loss)


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


print("SNR ", a)

print("160000train time %.1f %s" % (train, trainUnit))
print("40000test time %.1f %s" % (test, testUnit))

print("full test es mean", fullCapacity_Mean)

print("sub test es mean",Best_subCapacity_Mean)


print('accuracy', acc)
print('sub channel capacity mean', Capacity_Mean)


print('loss mean', Loss_Mean)

print('variance', Loss_Variance)


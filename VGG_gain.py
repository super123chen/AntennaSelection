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
import math

x = pd.read_csv(r'/home/wwj/chenqiliang/8822_list.csv', header=None)
matrix = x.values
def GetIndexFrom(y_pre):
    for i in range(0, 784):
        if y_pre == matrix[i][0]:
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i,4]

n_class = 784  
signal_power = 1.0 

a = 10
n = 8
X_orig = pd.read_csv(r"/home/wwj/chenqiliang/SVD/All-channel_matrix_p_1.csv").iloc[:, 1:]
X_orig = np.asarray(X_orig, np.float32)
dataset = X_orig.reshape(X_orig.shape[0],8,8,1)

label = pd.read_csv(r"/home/wwj/chenqiliang/SVD/gain_labels_p_1.csv").iloc[:, 1]

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




class VGG16(nn.Module):
    def __init__(self,n_class):
        super(VGG16, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            
        )
       
        self.classifier = nn.Sequential(
           
            nn.Linear(in_features=2*2*512,out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096,out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(in_features=4096,out_features=784) 
        )

    def forward(self,x):  
        x = x.reshape(-1, 1, 8, 8)
        x = self.features(x)
        
        
        x = torch.flatten(x,1)
        result = self.classifier(x)
        return result





















n_class = 784  
model = VGG16(n_class)












train_dataset = TensorDataset(torch.Tensor(xTrain), torch.Tensor(yTrain))
test_dataset = TensorDataset(torch.Tensor(xTest), torch.Tensor(yTest))

train_loader = DataLoader(train_dataset, batch_size=1280, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1280, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = VGG16(784).to(device)
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







     
          
          
          





model = VGG16(784)
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

































I = np.eye(8)
I2 = np.eye(2)
Loss = []
Gain = []
for i in range(40000):
    ArrayA = xTest_np[i].reshape(8, 8)
    ArrayA = np.matrix(ArrayA)

    i1, i2, j1, j2 = GetIndexFrom(ResNet_Pre_np1[i])
    Pre_sub = mat(np.zeros((2, 2)), dtype=float)
    Pre_sub[0, 0] = ArrayA[i1, j1]
    Pre_sub[0, 1] = ArrayA[i1, j2]
    Pre_sub[1, 0] = ArrayA[i2, j1]
    Pre_sub[1, 1] = ArrayA[i2, j2]
    Pre_fullGian = math.sqrt(1 / 2) * np.linalg.norm(ArrayA, ord='fro')
    Pre_subGian = math.sqrt(1 / 2) * np.linalg.norm(Pre_sub, ord='fro')
    Gain.append(Pre_subGian)
    Loss.append(Pre_fullGian - Pre_subGian)

Gain_Mean = np.mean(Gain)
Loss_Mean = np.mean(Loss)
Loss_Variance = np.var(Loss)


#print("acc is %.6f "%(acc))
#print(f"{acc:.1f}%")
#print("160000traintime%.1f %s" % (computation_time(train)[0], computation_time(train)[1]))
#print("40000testtime%.1f %s" % (computation_time(test)[0], computation_time(test)[1]))

print(Gain_Mean)
print(Loss_Mean)
print(Loss_Variance)

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



print("SNR",a)
print("160000train time %.1f %s" % (train, trainUnit))
print("40000 train time %.1f %s" % (test, testUnit))

print(f"{acc:.2f}%")






predicted_classes = torch.argmax(ResNet_Pre1, dim=1).numpy()  
true_classes = np.argmax(yTest[:40000], axis=1) 


def calculate_ber(predicted, true):
   
    predicted_binary = np.zeros((predicted.size, n_class), dtype=bool) 
    true_binary = np.zeros((true.size, n_class), dtype=bool)

    for i in range(predicted.size):
        predicted_binary[i, predicted[i]] = 1
        true_binary[i, true[i]] = 1

   
    errors = np.sum(predicted_binary != true_binary)

   
    total_bits = predicted_binary.size

   
    ber = errors / total_bits
    return ber

ber = calculate_ber(predicted_classes, true_classes)

print(f"Bit Error Rate (BER): {ber:.6f}")
















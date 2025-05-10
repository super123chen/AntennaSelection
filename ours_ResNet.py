from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
from torch.optim import AdamW
from uitls_8822_capacity import computation_time
import math
from time import time
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d
from torch.cuda.amp import GradScaler, autocast




from numpy import mat



x = pd.read_csv(r'/home/wwj/chenqiliang/8822_list.csv',header=None)
matrix = x.values



def GetIndexFrom(y_pre):
    for i in range(0, 784):
        if y_pre == matrix[i][0]:
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i, 4]


a=1
dataset = pd.read_csv(r'/home/wwj/chenqiliang/SVD/newAll-channel_matrix_p_1.csv').iloc[:, 1:]
dataset = np.asarray(dataset, np.float32)
dataset = dataset.reshape(dataset.shape[0], 8, 8, 1)
label = pd.read_csv(r'/home/wwj/chenqiliang/SVD/newcapacity_labels_p_1.csv').iloc[:, 1]
label = np.asarray(label, np.int32)
label.astype(np.int32)

#one hot
n_class = 784
n_sample = label.shape[0]
label_array = np.zeros((n_sample, n_class))
for i in range(n_sample):
    label_array[i, label[i] - 1] = 1



xTrain, xTest, yTrain, yTest = train_test_split(dataset, label_array, test_size=0.1, random_state=40)
print("xTrain: ", len(xTrain))
print(xTrain.shape)

print("xTest: ", len(xTest))





class SEChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction),
            nn.ReLU(),
            nn.Linear(channel//reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EnhancedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False):
        super().__init__()
        self.main = nn.Sequential(
            Conv2d(in_ch, out_ch, 3, 
                  stride=2 if downsample else 1, 
                  padding=1),
            BatchNorm2d(out_ch),
            SEChannelAttention(out_ch),  
            ReLU(),
            Conv2d(out_ch, out_ch, 1),    
            BatchNorm2d(out_ch),
            Shadow(out_ch)               
        )
        
        self.identity = nn.Sequential()
        if downsample or in_ch != out_ch:
            self.identity = nn.Sequential(
                Conv2d(in_ch, out_ch, 1, 
                      stride=2 if downsample else 1),
                Shadow(out_ch)
            )
            
    def forward(self, x):
        identity = self.identity(x)
        x = self.main(x)
        return F.relu(x + identity)

class Shadow(nn.Module):
    def __init__(self, inc):
        super(Shadow, self).__init__()
        self.lin1 = nn.Linear(int(inc / 4), int(inc / 4))
        self.lin2 = nn.Linear(int(inc / 4), int(inc / 4))
        self.lin3 = nn.Linear(int(inc / 4), int(inc / 4))
        self.lin4 = nn.Linear(int(inc / 4), int(inc / 4))
        self.conv = nn.Conv2d(int(2 * inc), inc, 1)
        

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  
        x_chunks = torch.chunk(x, chunks=4, dim=3)
        x_chunk1, x_chunk2, x_chunk3, x_chunk4 = x_chunks

        x_chunk1 = self.lin1(x_chunk1)
        x_chunk1 = F.relu(x_chunk1)
        x_chunk2 = self.lin2(x_chunk2)
        x_chunk2 = F.gelu(x_chunk2)
        x_chunk3 = self.lin3(x_chunk3)
        x_chunk3 = F.selu(x_chunk3)
        x_chunk4 = self.lin4(x_chunk4)
        x_chunk4 = x_chunk4*F.sigmoid(x_chunk4)
        

        x = torch.cat([x, x_chunk1, x_chunk2, x_chunk3, x_chunk4], dim=3)
        
       

        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)

       
        return x   



class ResNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        self.model0 = Sequential(
            Conv2d(1, 64, kernel_size=(3,3), padding=1),
            BatchNorm2d(64),
            Shadow(64),
            ReLU()
        )
        
       
        self.blocks = Sequential(
            EnhancedResBlock(64, 64),
            
            EnhancedResBlock(64, 64),
            
            EnhancedResBlock(64, 128, downsample=True),
           
            EnhancedResBlock(128, 128),
            
            EnhancedResBlock(128, 256, downsample=True),
            
            EnhancedResBlock(256, 256),
            
            ReLU(),
            EnhancedResBlock(256, 512, downsample=True),
            
            EnhancedResBlock(512, 512)
          
        )
        
        
        self.classifier = Sequential(
            AdaptiveAvgPool2d((1,1)),
            Flatten(),
            Linear(512, n_class)
        )

    def forward(self, x):
        x = x.reshape(-1, 1, 8, 8)
        x = self.model0(x)
        x = self.blocks(x)
        return self.classifier(x)




train_dataset = TensorDataset(torch.Tensor(xTrain), torch.Tensor(yTrain))
test_dataset = TensorDataset(torch.Tensor(xTest), torch.Tensor(yTest))

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ResNet(784).to(device)
#model.load_state_dict(torch.load(f"/home/wwj/chenqiliang/model.pth"))




lr = 0.001 
initial_lr=0.1
weight_decay = 0.0001  
betas = (0.95, 0.999)  
#eps = 1e-8
eps=1e-8
momentum=0.9
alpha=0.99
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)




criterion = nn.CrossEntropyLoss()
current_lr = optimizer.param_groups[0]['lr']
best_loss = float('inf') 


trainStart = time()
num_epochs = 20


 
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

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()




train = time() - trainStart



model = ResNet(784)
model.load_state_dict(torch.load(f"/home/wwj/chenqiliang/model.pth"))
device = torch.device('cpu')
model.to(device)
ResNet_Pre1 = model(torch.from_numpy(xTest[0:20000]).to(device))


########################################################333wo xiede
testStart=time()
pre_array1 = np.zeros((20000, n_class))
index1=torch.argmax(ResNet_Pre1,dim=1)
for i in range(20000):
    pre_array1[i,index1[i]] = 1
    
    
test = time() - testStart   

aaa1 = torch.argmax(ResNet_Pre1, axis=1) + 1
ResNet_Pre_np1 = aaa1.numpy()
ResNet_Pre_np1_ = pre_array1
ResNet_Pre_np_=ResNet_Pre_np1_





############################################################3















###############################################################################




xTest_np = np.array(xTest[0:20000])


##################dui yu ce chuli

label_array = np.zeros((n_sample, n_class))
for i in range(n_sample):
    label_array[i, label[i] - 1] = 1
###################
yTest_indices = np.array(yTest[:20000])





b=np.all(ResNet_Pre_np_ == yTest_indices, axis=1)

acc = np.sum(b) / 20000.0 * 100.0


I1 = np.eye(8)
I2 = np.eye(2)

Pre_Loss = []

Pre_Capacity = []
for i in range(20000):
    ArrayA = xTest_np[i].reshape(8, 8)
    ArrayA = np.matrix(ArrayA)

    i1, i2, j1, j2 = GetIndexFrom(ResNet_Pre_np1[i])  
    Pre_sub = ArrayA[[i1, i2]][:, [j1, j2]]
    Pre_fullCapacity = np.log2(np.linalg.det(I1 + a * ArrayA.T * ArrayA / 8))
    Pre_subCapacity= np.log2(np.linalg.det(I2 + a *  Pre_sub.T *  Pre_sub / 2))

    Pre_Capacity.append(Pre_subCapacity)
    Pre_Loss.append(Pre_fullCapacity - Pre_subCapacity)


Capacity_Mean = np.mean(Pre_Capacity)
Loss_Mean = np.mean(Pre_Loss)
Loss_Variance = np.var(Pre_Loss)


#print("acc is %.6f "%(acc))
print(f"{acc:.1f}%")
print("160000traintime%.1f %s" % (computation_time(train)[0], computation_time(train)[1]))
print("40000testtime%.1f %s" % (computation_time(test)[0], computation_time(test)[1]))

print(Capacity_Mean)




print(Loss_Mean)
print(Loss_Variance)





'''

I = np.eye(8)
I2 = np.eye(2)
Loss = []
Gain = []
for i in range(20000):
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

print("160000traintime%.1f %s" % (computation_time(train)[0], computation_time(train)[1]))
print("40000testtime%.1f %s" % (computation_time(test)[0], computation_time(test)[1]))

print(Gain_Mean)
print(Loss_Mean)
print(Loss_Variance)

print(f"{acc:.1f}%")

'''

















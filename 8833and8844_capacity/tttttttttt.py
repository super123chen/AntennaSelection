from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
from torch.optim import AdamW
from uitls_8833_capacity import computation_time
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







x = pd.read_csv(r'/home/wwj/chenqiliang/ceshiyongde/8833select.csv',header=None)
matrix = x.values



def GetIndexFrom(y_pre):
    for i in range(0, 3136):
        if y_pre == matrix[i][0]:
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i, 4], matrix[i, 5], matrix[i, 6]


a = 10






label_df = pd.read_csv(r'/home/wwj/chenqiliang/ceshiyongde/208833capacity_labels_p_10.csv').iloc[:, 1]
label_df = label_df.dropna()  

dataset_df = pd.read_csv(r'/home/wwj/chenqiliang/ceshiyongde/208833All-channel_matrix_p_10.csv').iloc[:, 1:]


dataset_df = dataset_df.loc[label_df.index]


label = label_df.values
dataset = dataset_df.values


dataset = np.asarray(dataset, np.float32)
dataset = dataset.reshape(dataset.shape[0], 8, 8, 1)


label = np.asarray(label, np.int32)

n_class = 3136
n_sample = label.shape[0]


label_array = np.zeros((n_sample, n_class))


for i in range(n_sample):
    label_array[i, label[i] - 1] = 1



























xTrain, xTest, yTrain, yTest = train_test_split(dataset, label_array, test_size=0.1, random_state=40)
print("xTrain: ", len(xTrain))
print(xTrain.shape)

print("xTest: ", len(xTest))


class ghost(nn.Module):
    def __init__(self, inc):
        super(ghost, self).__init__()
        self.lin1 = nn.Linear(int(inc / 4), int(inc / 4))
        self.lin2 = nn.Linear(int(inc / 4), int(inc / 4))
        self.lin3 = nn.Linear(int(inc / 4), int(inc / 4))
        self.lin4 = nn.Linear(int(inc / 4), int(inc / 4))
        self.conv = nn.Conv2d(int(2 * inc), inc, 1)
        

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # bhwc
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


class Resnet(nn.Module):
    def __init__(self, n_class):
        super(Resnet, self).__init__()
        self.model0 = Sequential(
           
            
            Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ghost(64),
            ReLU(),
          
        )
        self.R0 = ReLU()
        
        self.model1 = Sequential(
            
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ghost(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ghost(64),
            ReLU(),
        )

        self.R1 = ReLU()
        
        self.model2 = Sequential(
            
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ghost(64),
            ReLU(),
            
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ghost(64),
            ReLU(),
            
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
           
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ghost(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ghost(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ghost(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
           
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ghost(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ghost(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
           
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ghost(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ghost(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ghost(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ghost(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ghost(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
           
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ghost(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ghost(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ghost(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ghost(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ghost(512),
            ReLU(),
        )
        self.R8 = ReLU()

        
        self.aap = AdaptiveAvgPool2d((1, 1))
       
        self.flatten = Flatten(start_dim=1)
        
        self.fc = Linear(512, n_class)

    def forward(self, x):
        x = x.reshape(-1, 1, 8, 8)
        x = self.model0(x)

        f1 = x
        x = self.model1(x)
        x = x + f1
        x = self.R1(x)

        f1_1 = x
        x = self.model2(x)
        x = x + f1_1
        x = self.R2(x)

        f2_1 = x
        f2_1 = self.en1(f2_1)
        x = self.model3(x)
        x = x + f2_1
        x = self.R3(x)

        f2_2 = x
        x = self.model4(x)
        x = x + f2_2
        x = self.R4(x)

        f3_1 = x
        f3_1 = self.en2(f3_1)
        x = self.model5(x)
        x = x + f3_1
        x = self.R5(x)

        f3_2 = x
        x = self.model6(x)
        x = x + f3_2
        x = self.R6(x)

        f4_1 = x
        f4_1 = self.en3(f4_1)
        x = self.model7(x)
        x = x + f4_1
        x = self.R7(x)

        f4_2 = x
        x = self.model8(x)
        x = x + f4_2
        x = self.R8(x)

        
        x = self.aap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x




train_dataset = TensorDataset(torch.Tensor(xTrain), torch.Tensor(yTrain))
test_dataset = TensorDataset(torch.Tensor(xTest), torch.Tensor(yTest))

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Resnet(3136).to(device)
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

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()




train = time() - trainStart



model = Resnet(3136)
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
I2 = np.eye(3)

Pre_Loss = []

Pre_Capacity = []
for i in range(20000):
    ArrayA = xTest_np[i].reshape(8, 8)
    ArrayA = np.matrix(ArrayA)

    i1, i2,i3, j1, j2,j3 = GetIndexFrom(ResNet_Pre_np1[i])  
    Pre_sub = ArrayA[[i1, i2,i3]][:, [j1, j2,j3]]
    Pre_fullCapacity = np.log2(np.linalg.det(I1 + a * ArrayA.T * ArrayA / 8))
    Pre_subCapacity= np.log2(np.linalg.det(I2 + a *  Pre_sub.T *  Pre_sub / 3))

    Pre_Capacity.append(Pre_subCapacity)
    Pre_Loss.append(Pre_fullCapacity - Pre_subCapacity)


Capacity_Mean = np.mean(Pre_Capacity)
Loss_Mean = np.mean(Pre_Loss)
Loss_Variance = np.var(Pre_Loss)


#print("acc is %.6f "%(acc))
print(f"{acc:.1f}%")
print("180000traintime%.1f %s" % (computation_time(train)[0], computation_time(train)[1]))
print("20000testtime%.1f %s" % (computation_time(test)[0], computation_time(test)[1]))

print(Capacity_Mean)
print(Loss_Mean)
print(Loss_Variance)


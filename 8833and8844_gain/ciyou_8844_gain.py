from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
from torch.optim import AdamW
from uitls_8844_gain import computation_time
import math
from time import time
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import torch.nn.functional as F





x = pd.read_csv(r'/home/wwj/chenqiliang/ceshiyongde/8844select.csv',header=None)
matrix = x.values



def GetIndexFrom(y_pre):
    for i in range(0, 4900):
        if y_pre == matrix[i][0]:
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i, 4], matrix[i, 5], matrix[i, 6],matrix[i, 7],matrix[i, 8]


a = 10






label_df = pd.read_csv(r'/home/wwj/chenqiliang/8833and8844shuju/8844JTRAS_gain_label_10.csv').iloc[:, 1]
label_df = label_df.dropna()  

dataset_df = pd.read_csv(r'/home/wwj/chenqiliang/ceshiyongde/208833All-channel_matrix_p_10.csv').iloc[:, 1:]


dataset_df = dataset_df.loc[label_df.index]


label = label_df.values
dataset = dataset_df.values


dataset = np.asarray(dataset, np.float32)
dataset = dataset.reshape(dataset.shape[0], 8, 8, 1)


label = np.asarray(label, np.int32)

n_class = 4900
n_sample = label.shape[0]


label_array = np.zeros((n_sample, n_class))


for i in range(n_sample):
    label_array[i, label[i] - 1] = 1




xTrain, xTest, yTrain, yTest = train_test_split(dataset, label_array, test_size=0.1, random_state=40)
print("xTrain: ", len(xTrain))
print(xTrain.shape)

print("xTest: ", len(xTest))





        
        







class ResidualBottleneck(nn.Module):
    def __init__(self, in_channels, bottle_channels,out_channels, stride=1):
        super(ResidualBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, bottle_channels, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(bottle_channels)
        self.conv2 = nn.Conv2d(bottle_channels, bottle_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(bottle_channels)
        self.conv3 = nn.Conv2d(bottle_channels, out_channels, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.inc=in_channels
        self.outc=out_channels
        
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
    

        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.inc!=self.outc:
          residual = self.downsample(x)
        
       
        out += residual
        out = self.relu(out)
        return out












class Shadow(nn.Module):
    def __init__(self, inc):
        super(Shadow, self).__init__()
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




      
      
        
        
        








class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()

        self.convseq=nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=2, bias=True),
        Shadow(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=2, bias=True),
        Shadow(128),
        nn.ReLU6(inplace=True),
        nn.Conv2d(128, 256, kernel_size=2, bias=True),
        Shadow(256),
        nn.SELU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=2, bias=True),
        Shadow(256),
        nn.GELU(),
        nn.Conv2d(256, 128, kernel_size=2, bias=True),
        Shadow(128),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(128, 64, kernel_size=2,bias=True),
        Shadow(64),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
      
        
        )
        
        self.resblocks = nn.Sequential(
            ResidualBottleneck(64,16 ,64),         
            ResidualBottleneck(64, 32,128),            
            ResidualBottleneck(128, 32,128),         
            ResidualBottleneck(128, 64,256),            
            ResidualBottleneck(256, 64,256),
            
        )
      
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(256)
      
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)# wojiade         
        self.linear = nn.Linear(256, 4900)
        

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # bchw
        x=self.convseq(x)
        
        
        x = self.resblocks(x)
        x = self.relu(self.bn(x))
        x = self.avg_pool(x)
        
        x_flattened = x.flatten(start_dim=1)  # bchw
        x = self.dropout(x)#  wojiade

        x = self.linear(x_flattened)
        return x




























train_dataset = TensorDataset(torch.Tensor(xTrain), torch.Tensor(yTrain))
test_dataset = TensorDataset(torch.Tensor(xTest), torch.Tensor(yTest))

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = MyNetwork().to(device)
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
num_epochs = 60



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




model = MyNetwork()
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











I = np.eye(8)
I2 = np.eye(4)
Loss = []
Gain = []
for i in range(20000):
    ArrayA = xTest_np[i].reshape(8, 8)
    ArrayA = np.matrix(ArrayA)

    i1, i2,i3,i4, j1, j2,j3,j4 = GetIndexFrom(ResNet_Pre_np1[i])
    Pre_sub = ArrayA[[i1, i2,i3,i4]][:, [j1, j2,j3,j4]]
    Pre_fullGian = math.sqrt(1 / 4) * np.linalg.norm(ArrayA, ord='fro')
    Pre_subGian = math.sqrt(1 / 4) * np.linalg.norm(Pre_sub, ord='fro')
    Gain.append(Pre_subGian)
    Loss.append(Pre_fullGian - Pre_subGian)

Gain_Mean = np.mean(Gain)
Loss_Mean = np.mean(Loss)
Loss_Variance = np.var(Loss)


#print("acc is %.6f "%(acc))
print(f"{acc:.1f}%")
print("100000traintime%.1f %s" % (computation_time(train)[0], computation_time(train)[1]))
print("20000testtime%.1f %s" % (computation_time(test)[0], computation_time(test)[1]))

print(Gain_Mean)
print(Loss_Mean)
print(Loss_Variance)


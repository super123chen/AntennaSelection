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

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from numpy import mat


x = pd.read_csv(r'/home/wwj/chenqiliang/8822_list.csv',header=None)
matrix = x.values



def GetIndexFrom(y_pre):
    for i in range(0, 784):
        if y_pre == matrix[i][0]:
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i, 4]


a = 1
dataset = pd.read_csv(r'/home/wwj/chenqiliang/SVD/All-channel_matrix_p_1.csv').iloc[:, 1:]

dataset = np.asarray(dataset, np.float32)
dataset = dataset.reshape(dataset.shape[0], 8, 8, 1)
label = pd.read_csv(r'/home/wwj/chenqiliang/SVD/capacity_labels_p_1.csv').iloc[:, 1]
label = np.asarray(label, np.int32)
label.astype(np.int32)

#one hot
n_class = 784
n_sample = label.shape[0]
label_array = np.zeros((n_sample, n_class))
for i in range(n_sample):
    label_array[i, label[i] - 1] = 1



xTrain, xTest, yTrain, yTest = train_test_split(dataset, label_array, test_size=0.2, random_state=40)
print("xTrain: ", len(xTrain))
print(xTrain.shape)

print("xTest: ", len(xTest))







'''
class InceptionBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(InceptionBlock, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)

        self.branch3x3_1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.branch3x3_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)

        self.branch5x5_1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.branch5x5_2 = nn.Conv2d(out_channel, out_channel, kernel_size=5, padding=2, bias=False)

        self.branch_pool = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(4 * out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv1x1 = nn.Conv2d(in_channels=out_channel * 4, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        out = torch.cat(outputs, 1)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv1x1(out)

        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
'''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                      kernel_size=3, stride=stride, padding=1, bias=False)
        #self.conv1 = nn.Sequential(
             #InceptionBlock(in_channel=in_channel, out_channel=in_channel),
             #nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                       #kernel_size=3, stride=stride, padding=1, bias=False),
         #)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                      kernel_size=3, stride=1, padding=1, bias=False)
        #self.conv2 = nn.Sequential(
             #InceptionBlock(in_channel=out_channel, out_channel=out_channel),
             #nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                       #kernel_size=3, stride=1, padding=1, bias=False)
         #)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        #self.se1 = SEBlock(in_channel)
        #self.se2 = SEBlock(out_channel)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.se1(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # print(out.shape)
        out = self.bn2(out)

        #out = self.se2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=784,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        

        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(-1, 1, 8, 8)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x




def create_epsanet():
    layers = [2, 2, 2, 2]  
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=784)  
    return model




'''
if __name__ == '__main__':
    model = resnet34()
    input = torch.randn(16, 1, 8,8)
    
    out = model(input)
    print(model)
    print(out.shape)
'''   

train_dataset = TensorDataset(torch.Tensor(xTrain), torch.Tensor(yTrain))
test_dataset = TensorDataset(torch.Tensor(xTest), torch.Tensor(yTest))

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#model = ResNet(784).to(device)
#model= ResNet(BasicBlock, [3, 4, 23, 3]).to(device)
#model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=784).to(device) 
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
num_epochs = 30


 
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



#model= ResNet(BasicBlock, [3, 4, 23, 3]).to(device)
#model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=784).to(device) 
model.load_state_dict(torch.load(f"/home/wwj/chenqiliang/model.pth"))
device = torch.device('cpu')
model.to(device)
ResNet_Pre1 = model(torch.from_numpy(xTest[0:40000]).to(device))

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





############################################################3







###############################################################################




xTest_np = np.array(xTest[0:40000])


##################dui yu ce chuli

label_array = np.zeros((n_sample, n_class))
for i in range(n_sample):
    label_array[i, label[i] - 1] = 1
###################
yTest_indices = np.array(yTest[:40000])










predicted_classes = torch.argmax(ResNet_Pre1, dim=1).numpy() 
true_classes = np.argmax(yTest[:40000], axis=1)  

accuracy = np.sum(predicted_classes == true_classes) / len(true_classes) * 100  
precision = precision_score(true_classes, predicted_classes, average='macro')  
recall = recall_score(true_classes, predicted_classes, average='macro')  
f1 = f1_score(true_classes, predicted_classes, average='macro')  
conf_matrix = confusion_matrix(true_classes, predicted_classes)  



print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)


'''

b=np.all(ResNet_Pre_np_ == yTest_indices, axis=1)

acc = np.sum(b) / 40000.0 * 100.0
'''
'''
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
print("160000traintime%.1f %s" % (computation_time(train)[0], computation_time(train)[1]))
print("40000testtime%.1f %s" % (computation_time(test)[0], computation_time(test)[1]))

print(Gain_Mean)
print(Loss_Mean)
print(Loss_Variance)
 
'''   



I1 = np.eye(8)
I2 = np.eye(2)

Pre_Loss = []

Pre_Capacity = []
for i in range(40000):
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
#print(f"{acc:.1f}%")
print("160000traintime%.1f %s" % (computation_time(train)[0], computation_time(train)[1]))
print("40000testtime%.1f %s" % (computation_time(test)[0], computation_time(test)[1]))

print(Capacity_Mean)
print(Loss_Mean)
print(Loss_Variance)
    

    
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


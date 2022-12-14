import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import numpy as np
import matplotlib.pyplot as plt
from environments import RPD
from agent import RuleAgent
from ga import SimpleGA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

sum = 0
epi = 1
N = 60
for i in range(epi):
    ga = SimpleGA(N)
    val,pool = ga.evolve()
    sum += val
print(sum/epi)

class SimpleNet(nn.Module):
    def __init__(self,input_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc =nn.Linear(input_size, output_size)
                           
    def forward(self, x):
        output = self.fc(x)
        return F.log_softmax(output,dim=-1)

    
def train_excorder(data, t,batch_size=8):
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, t),
        batch_size=batch_size,
        shuffle=True,
    )
    return train_loader

def train(model,dataset,optimizer):
    model.train()
    losses=[]
    for data,target in dataset:
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target.squeeze_().long())
        loss.backward()
        optimizer.step()
        losses.append(loss)
    print(torch.tensor(losses).mean())
    
look_list = [pool[j].look_gene for j in range(N)]
character_list = [pool[i].character_val for i in range(N)]
#学習データ（動作確認用）
x = torch.tensor(look_list).float()
#教師データ（動作確認用）
t = torch.tensor(character_list).float()

#pytorchで訓練するためにデータを変換
dataset = train_excorder(x,t)
#モデル定義
model = SimpleNet(7,2)
#確率勾配法
optimizer = optim.SGD(model.parameters(), lr=0.01)

#エポック数
epochs=15

for epoch in range(epochs):
    train(model,dataset,optimizer)




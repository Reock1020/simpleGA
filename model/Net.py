import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


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
    


#学習データ（動作確認用）
x = torch.tensor(np.random.randn(100,7)).float()
#教師データ（動作確認用）
t = torch.tensor(np.zeros_like(np.random.randn(100,1))).float()

#pytorchで訓練するためにデータを変換
dataset = train_excorder(x,t)
#モデル定義
model = SimpleNet(7,2)
#確率勾配法
optimizer = optim.SGD(model.parameters(), lr=0.01)

#エポック数
epochs=10

for epoch in range(epochs):
    train(model,dataset,optimizer)
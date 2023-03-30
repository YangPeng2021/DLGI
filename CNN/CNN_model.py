import torch
import torch.optim as optim
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1, 8,3,1,1,bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True))
        
        self.pool = nn.Sequential(nn.MaxPool2d(2))
        
        self.block2 = nn.Sequential(nn.Conv2d(8, 16,3,1,1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True))
        
        self.block3 = nn.Sequential(nn.Flatten(),
            nn.Linear(16*8*8, 1024))
       
        
    def forward(self,x):
        #the first convolutional layer
        layer_1=self.block1(x)
        #the first pooling layer        
        layer_1=self.pool(layer_1)

        #the second convolutional layer        
        layer_2=self.block2(layer_1)
        #the second pooling layer
        layer_2=self.pool(layer_2)

        #the fully connected layer       
        layer_3=self.block3(layer_2)
        layer_3=torch.sigmoid(layer_3)
        #print(layer_3.shape)
        
        out=torch.reshape(layer_3,(20,1,32,32))
 
        #print(out.shape)
        return out




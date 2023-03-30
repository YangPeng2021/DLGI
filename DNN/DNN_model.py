import torch
import torch.optim as optim
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self,n_hiddens=3,drop_out_rate=0.2):
        super(DNN, self).__init__()
          
        self.resize_layer=nn.Sequential(nn.Flatten())
        
        layers=[]
        for i in range(n_hiddens):
            layers.append(nn.Linear(1024,1024))
            layers.append(nn.BatchNorm1d(1024))
            layers.append(nn.ReLU(True))
            layers.append(nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity())
             
        self.hidden_layer = nn.Sequential(*layers)    
        
        self.output_layer=nn.Sequential(nn.Linear(1024,1024),
                         nn.Sigmoid())
 
    def forward(self,x):
        #resize layer
        resize_x=self.resize_layer(x)
        
        #hidden layer
        hidden_x=self.hidden_layer(resize_x)
        
        #output_layer
        out=self.output_layer(hidden_x)
        
        out=torch.reshape(out,(20,1,32,32))
 
        #print(out.shape)
        return out




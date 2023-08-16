# https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

# same conv on first and second images
# transform on tensors
# mlp

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self, shape, batch_size):
        
        super().__init__()
        
        self.shape = shape
        self.batch_size = batch_size
        
        kernel_size = 3
        conv_channels = 8
        
        attention_embed = conv_channels
        attention_heads = 1
        
        fc_channels = 64
        
        self.conv1 = nn.Conv2d(in_channels=shape[2], out_channels=conv_channels, kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=kernel_size, padding='same')
        
        self.attention  = nn.MultiheadAttention(attention_embed, num_heads=attention_heads, batch_first=True)
        
        self.fc1 = nn.Linear(shape[0] * shape[1] * conv_channels, fc_channels)
#         self.fc1 = nn.Linear(shape[0] * shape[1] * shape[2], fc_channels)
        self.fc2 = nn.Linear(fc_channels, 1)
        
    def forward(self, x):
        
#         print('0', x.shape)
        
        x = F.relu(self.conv1(x)) 
#         print('1', x.shape)
        
        x = F.relu(self.conv2(x))
#         print('2', x.shape)
        
#         x = torch.flatten(x, start_dim=2)
#         print('3', x.shape)
        
#         x = x.permute(0, 2, 1)
#         print('4', x.shape)
        
#         x, _ = self.attention(x, x, x)
#         print('5', x.shape)
        
        x = torch.flatten(x, start_dim=1)
#         print('6', x.shape)
        
        x = F.relu(self.fc1(x))
#         print('7', x.shape)
        
        x = self.fc2(x)
#         print('8', x.shape)

#         x = torch.reshape(x, (x.shape[0], self.shape[0], self.shape[1], self.shape[2]))
#         print('9', x.shape)
        
        return x
    
def model_summary(model):

    print("Model Summary")
    print()
    print("Layer_name"+"\t"*7+"Number of Parameters")
    print("="*100)
    model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
    layer_name = [child for child in model.children()]
    j = 0
    total_params = 0
    print("\t"*10)
    for i in layer_name:
        print()
        param = 0
        try:
            bias = (i.bias is not None)
        except:
            bias = False  
        if not bias:
            param =model_parameters[j].numel()+model_parameters[j+1].numel()
            j = j+2
        else:
            param =model_parameters[j].numel()
            j = j+1
        print(str(i)+"\t"*3+str(param))
        total_params+=param
    print("="*100)
    print(f"Total Params:{total_params}")       



class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(16)
        self.drop=nn.Dropout(p=0.15)        
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(2, 2, 2), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        
        return out


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 320),
            nn.LeakyReLU(),
            nn.BatchNorm1d(320),
            nn.Dropout(p=0.15),

            nn.Linear(320, 360),
            nn.LeakyReLU(),
            nn.BatchNorm1d(360),
            nn.Dropout(p=0.15),

            nn.Linear(360, 620),
            nn.LeakyReLU(),
            nn.BatchNorm1d(620),
            nn.Dropout(p=0.15),

            nn.Linear(620, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.15),

            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = x.view(x.shape[0],-1)
        return self.model(x)

if __name__ == "__main__":
    model = MLP(1000)
    x = torch.rand(3, 1, 10, 10, 10)
    print(model(x).shape)